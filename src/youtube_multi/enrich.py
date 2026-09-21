"""Per-frame enrichment: OCR, perceptual-hash dedupe, kind classification, diffs."""
from __future__ import annotations

import difflib
import os
import platform
import re
import sys
from pathlib import Path
from shutil import which

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

from .models import KINDS, Diff, Scene

# --- tesseract discovery -------------------------------------------------------


def _tesseract_fallbacks() -> list[Path]:
    system = platform.system()
    if system == "Windows":
        return [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        ]
    if system == "Darwin":
        return [
            Path("/opt/homebrew/bin/tesseract"),
            Path("/usr/local/bin/tesseract"),
        ]
    return [
        Path("/usr/bin/tesseract"),
        Path("/usr/local/bin/tesseract"),
    ]


_INSTALL_HINT = {
    "Windows": "winget install UB-Mannheim.TesseractOCR",
    "Darwin": "brew install tesseract",
}


def configure_tesseract() -> None:
    """Point pytesseract at a tesseract binary.

    Order: $TESSERACT_CMD, then PATH, then per-platform install locations.
    """
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        if Path(env_cmd).is_file():
            pytesseract.pytesseract.tesseract_cmd = env_cmd
            return
        raise RuntimeError(f"TESSERACT_CMD={env_cmd!r} is not a file")
    if which("tesseract"):
        return
    for candidate in _tesseract_fallbacks():
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return
    hint = _INSTALL_HINT.get(platform.system(), "apt install tesseract-ocr (or your distro's equivalent)")
    raise RuntimeError(
        "tesseract not found on PATH or in known install locations. "
        f"Install with: {hint}  — or set TESSERACT_CMD to the binary."
    )


# --- OCR -------------------------------------------------------------------------


def ocr_frame(image_path: Path, min_chars: int) -> str:
    """Whitespace-collapsed OCR text (the original Phase 0 behaviour)."""
    try:
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"  [ocr-warn] {image_path.name}: {e}", file=sys.stderr)
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= min_chars else ""


def ocr_scene(scene: Scene, min_chars: int) -> None:
    """Populate scene.ocr, scene.ocr_lines, scene.ocr_confidence and the OCR-derived
    layout features from a single tesseract pass (image_to_data)."""
    try:
        with Image.open(scene.image_path) as img:
            data = pytesseract.image_to_data(img, output_type=Output.DICT)
    except Exception as e:
        print(f"  [ocr-warn] {scene.image_path.name}: {e}", file=sys.stderr)
        return

    lines: dict[tuple[int, int, int], list[tuple[int, str, float, int]]] = {}
    for i, raw in enumerate(data["text"]):
        word = raw.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((data["left"][i], word, conf, data["width"][i]))

    ocr_lines: list[str] = []
    confs: list[float] = []
    char_widths: list[float] = []
    line_lefts: list[int] = []
    for key in sorted(lines):
        words = sorted(lines[key])
        ocr_lines.append(" ".join(w for _, w, _, _ in words))
        line_lefts.append(words[0][0])
        for _, w, c, width in words:
            confs.append(c)
            if len(w) >= 3 and width > 0:
                char_widths.append(width / len(w))

    full = " ".join(ocr_lines)
    scene.ocr = full if len(full) >= min_chars else ""
    scene.ocr_lines = ocr_lines if len(full) >= min_chars else []
    scene.ocr_confidence = round(float(np.mean(confs)), 1) if confs else 0.0

    # Layout features for the classifier.
    f = scene.features
    f["n_lines"] = float(len(scene.ocr_lines))
    f["n_chars"] = float(len(full)) if scene.ocr_lines else 0.0
    if len(char_widths) >= 4:
        cw = np.asarray(char_widths)
        f["char_width_cv"] = float(cw.std() / cw.mean()) if cw.mean() > 0 else 1.0
        unit = float(np.median(cw))
        levels = {int(round(left / max(unit, 1.0))) for left in line_lefts}
        f["indent_levels"] = float(len(levels))
    else:
        f["char_width_cv"] = 1.0
        f["indent_levels"] = 0.0


# --- perceptual hashing ------------------------------------------------------------


def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in bits.flatten():
        out = (out << 1) | int(bool(b))
    return out


def phash(img: np.ndarray) -> int:
    """64-bit DCT perceptual hash (same construction as imagehash.phash)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    return _bits_to_int(low > np.median(low))


def dhash(img: np.ndarray) -> int:
    """64-bit difference hash (horizontal gradient of a 9x8 thumbnail)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def hash_scenes(scenes: list[Scene]) -> None:
    for s in scenes:
        img = cv2.imread(str(s.image_path))
        if img is None:
            continue
        s.phash = phash(img)
        s.dhash = dhash(img)


def dedupe_frames(scenes: list[Scene], max_distance: int | None, duration: float) -> int:
    """Mark near-duplicate consecutive frames as dropped and fill visible ranges.

    Each frame is initially visible from its own time to the next frame's time
    (or the video end). When a frame is within ``max_distance`` (pHash hamming)
    of the current survivor, it is marked ``duplicate`` and the survivor's
    ``visible_to`` is extended over it. ``max_distance=None`` disables dedupe.
    Returns the number of duplicates.
    """
    n = len(scenes)
    for i, s in enumerate(scenes):
        s.visible_from = s.t_seconds
        s.visible_to = scenes[i + 1].t_seconds if i + 1 < n else max(duration, s.t_seconds)
    if max_distance is None:
        return 0
    dupes = 0
    survivor: Scene | None = None
    for s in scenes:
        if survivor is not None and hamming(survivor.phash, s.phash) <= max_distance:
            s.dropped_reason = "duplicate"
            s.duplicate_of = survivor.idx
            survivor.visible_to = s.visible_to
            dupes += 1
            continue
        survivor = s
    return dupes


# --- classification -----------------------------------------------------------------

_CASCADE: cv2.CascadeClassifier | None = None


def _face_cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


_CODE_TOKEN_RE = re.compile(
    r"(\bdef |\bclass |\bimport |\bfrom \w+ import|\breturn\b|\bconst |\blet |\bvar |\bfunc |\bfn |"
    r"=>|==|!=|\{|\}|\[\]|\(\)|;\s*$|^\s*#include|^\s*//|^\s*/\*|\bif\s*\(|\bfor\s*\(|\bwhile\s*\(|"
    r"\bself\.|\bthis\.|\bnull\b|\bNone\b|\btrue\b|\bfalse\b|\bTrue\b|\bFalse\b|</?\w+>)"
)
_PROMPT_RE = re.compile(r"^(\$|>|❯|»|%|#|~|PS [A-Z]:\\|[\w.-]+@[\w.-]+:|[\w~/.-]*\$)\s")


def image_features(img: np.ndarray) -> dict[str, float]:
    """Cheap visual features: faces, luminance/saturation stats, edges, lines, boxes."""
    h, w = img.shape[:2]
    area = float(h * w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    f: dict[str, float] = {}
    f["dark_frac"] = float((v < 60).mean())
    f["light_frac"] = float((v > 200).mean())
    f["mean_sat"] = float(s.mean())
    f["sat_frac"] = float((s > 80).mean())

    # Faces (frontal Haar), scaled to keep detection cheap and stable.
    scale = 480.0 / max(h, 1)
    small = cv2.resize(gray, (int(w * scale), 480)) if scale < 1 else gray
    faces = _face_cascade().detectMultiScale(
        small, scaleFactor=1.1, minNeighbors=5, minSize=(small.shape[0] // 10, small.shape[0] // 10)
    )
    f["n_faces"] = float(len(faces))
    f["face_frac"] = float(max((fw * fh for _, _, fw, fh in faces), default=0) / float(small.shape[0] * small.shape[1]))

    edges = cv2.Canny(gray, 80, 200)
    f["edge_density"] = float((edges > 0).mean())

    min_len = int(0.15 * max(h, w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_len, maxLineGap=6)
    f["n_hough_lines"] = float(0 if lines is None else len(lines))

    # Boxes: regions of the foreground/background mask whose bounding box is
    # mostly filled and whose outline is nearly polygonal. Connector lines add a
    # vertex or two, so we allow 4-6 and lean on the fill ratio instead.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    seen: set[tuple[int, int, int, int]] = set()
    for c in contours:
        a = cv2.contourArea(c)
        if a < 0.005 * area or a > 0.6 * area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if a / float(bw * bh) < 0.75:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if not 4 <= len(approx) <= 6:
            continue
        seen.add((x // 8, y // 8, bw // 8, bh // 8))  # inner/outer outline contours collapse
    f["n_rects"] = float(len(seen))
    f["megapixels"] = area / 1e6
    return f


def text_features(ocr_lines: list[str]) -> dict[str, float]:
    n = max(len(ocr_lines), 1)
    code_hits = sum(1 for line in ocr_lines if _CODE_TOKEN_RE.search(line))
    prompt_hits = sum(1 for line in ocr_lines if _PROMPT_RE.match(line))
    return {"code_ratio": code_hits / n, "prompt_ratio": prompt_hits / n}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def classify_features(f: dict[str, float]) -> tuple[str, float]:
    """Rule table over features -> (kind, confidence). Pure, testable."""
    g = f.get
    n_lines = g("n_lines", 0.0)
    chars_per_mp = g("n_chars", 0.0) / max(g("megapixels", 1.0), 0.05)
    text_amt = _clamp(chars_per_mp / 400.0)  # ~400 chars per megapixel = dense text
    mono = _clamp((0.22 - g("char_width_cv", 1.0)) / 0.15) if n_lines >= 3 else 0.0
    code_r = _clamp(g("code_ratio", 0.0) / 0.3)
    prompt_r = _clamp(g("prompt_ratio", 0.0) / 0.2)
    indent = 1.0 if g("indent_levels", 0.0) >= 3 else 0.5
    dark = g("dark_frac", 0.0)
    light = g("light_frac", 0.0)
    has_face = g("n_faces", 0.0) >= 1

    scores = {
        "talking-head": (_clamp(g("face_frac", 0.0) / 0.03) * (1.0 - 0.6 * text_amt)) if has_face else 0.0,
        # Dark + monospace text; prompts are the strongest signal. A dark editor
        # full of code tokens but no prompts should lose to "code".
        "terminal": _clamp(dark / 0.55) * _clamp(n_lines / 4.0) * (0.35 + 0.65 * max(0.6 * mono, prompt_r)) * (1.0 - 0.5 * code_r * (1.0 - prompt_r)),
        "code": _clamp(n_lines / 6.0) * max(mono * indent, code_r) * (1.0 - 0.5 * prompt_r),
        "slide": text_amt * (1.0 - _clamp(dark / 0.6)) * (1.0 - 0.6 * max(mono, code_r)) * (1.0 - 0.5 * _clamp(g("face_frac", 0.0) / 0.03)),
        "diagram": (0.6 * _clamp(g("n_rects", 0.0) / 4.0) + 0.4 * _clamp(g("n_hough_lines", 0.0) / 20.0)) * (1.0 - 0.5 * text_amt),
        # Bright, unsaturated, some strokes but little OCR-able text and no boxes.
        "whiteboard": _clamp(light / 0.7) * (1.0 - _clamp(g("mean_sat", 0.0) / 60.0)) * _clamp((g("edge_density", 0.0) - 0.015) / 0.03) * (1.0 - text_amt) * (1.0 - _clamp(g("n_rects", 0.0) / 3.0)),
        "other": 0.15,
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (kind, best), (_, second) = ranked[0], ranked[1]
    if best <= 0.15:
        return "other", 0.3
    confidence = _clamp(best) * (0.5 + 0.5 * _clamp((best - second) / max(best, 1e-6)))
    return kind, round(confidence, 2)


def classify_scene(scene: Scene) -> None:
    img = cv2.imread(str(scene.image_path))
    if img is None:
        scene.kind, scene.kind_confidence = "other", 0.0
        return
    scene.features.update(image_features(img))
    scene.features.update(text_features(scene.ocr_lines))
    scene.kind, scene.kind_confidence = classify_features(scene.features)


def parse_drop_kinds(values: list[str] | None) -> list[str]:
    if values is None:
        return ["talking-head"]
    kinds: list[str] = []
    for v in values:
        for part in v.split(","):
            part = part.strip().lower()
            if not part or part == "none":
                continue
            if part not in KINDS:
                raise ValueError(f"unknown --drop kind {part!r}; choose from {', '.join(KINDS)} or none")
            kinds.append(part)
    return kinds


def drop_by_kind(scenes: list[Scene], kinds: list[str]) -> int:
    dropped = 0
    for s in scenes:
        if s.kept and s.kind in kinds:
            s.dropped_reason = f"kind:{s.kind}"
            dropped += 1
    return dropped


def remove_dropped_files(scenes: list[Scene], frames_dir: Path, keep: bool) -> None:
    dropped_dir = frames_dir / "dropped"
    for s in scenes:
        if s.kept or not s.image_path.exists():
            continue
        if keep:
            dropped_dir.mkdir(parents=True, exist_ok=True)
            target = dropped_dir / s.image_path.name
            s.image_path.replace(target)
            s.image_path = target
        else:
            s.image_path.unlink()


# --- diffs ----------------------------------------------------------------------------


def pixel_diff_region(prev: np.ndarray, cur: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
    """Bounding box (x, y, w, h) of the changed area between two frames and the
    fraction of pixels that changed. Returns (None, 0.0) if nothing changed."""
    if prev.shape != cur.shape:
        cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]))
    a = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    d = cv2.absdiff(a, b)
    _, mask = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
    changed_frac = float((mask > 0).mean())
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1), changed_frac


def diff_scenes(prev: Scene, cur: Scene, diffs_dir: Path, crop_max_frac: float = 0.6) -> Diff:
    diff = Diff(vs_idx=prev.idx)
    stem = f"{prev.idx:04d}_{cur.idx:04d}"

    if prev.ocr_lines or cur.ocr_lines:
        udiff = list(
            difflib.unified_diff(
                prev.ocr_lines, cur.ocr_lines,
                fromfile=f"frame {prev.idx:04d}", tofile=f"frame {cur.idx:04d}", lineterm="",
            )
        )
        diff.added = sum(1 for l in udiff if l.startswith("+") and not l.startswith("+++"))
        diff.removed = sum(1 for l in udiff if l.startswith("-") and not l.startswith("---"))
        if diff.added or diff.removed:
            diffs_dir.mkdir(parents=True, exist_ok=True)
            diff.text_path = diffs_dir / f"{stem}.txt"
            diff.text_path.write_text("\n".join(udiff) + "\n", encoding="utf-8")

    a = cv2.imread(str(prev.image_path))
    b = cv2.imread(str(cur.image_path))
    if a is not None and b is not None:
        region, frac = pixel_diff_region(a, b)
        diff.changed_frac = round(frac, 4)
        if region is not None:
            x, y, w, h = region
            H, W = b.shape[:2]
            if w * h < crop_max_frac * W * H:
                pad = 8
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
                diff.changed_region = (x0, y0, x1 - x0, y1 - y0)
                diffs_dir.mkdir(parents=True, exist_ok=True)
                diff.image_path = diffs_dir / f"{stem}.jpg"
                cv2.imwrite(str(diff.image_path), b[y0:y1, x0:x1], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            else:
                diff.changed_region = (x, y, w, h)
    return diff


def diff_consecutive(scenes: list[Scene], diffs_dir: Path) -> int:
    """Attach a Diff to each kept frame whose previous kept frame has the same kind."""
    prev: Scene | None = None
    count = 0
    for s in scenes:
        if not s.kept:
            continue
        if prev is not None and prev.kind == s.kind:
            s.diff = diff_scenes(prev, s, diffs_dir)
            count += 1
        prev = s
    return count
