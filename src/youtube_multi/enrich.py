"""Per-frame enrichment: OCR (and, in later phases, dedupe/classify/diff)."""
from __future__ import annotations

import os
import platform
import re
import sys
from pathlib import Path
from shutil import which

import pytesseract
from PIL import Image


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


def ocr_frame(image_path: Path, min_chars: int) -> str:
    try:
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"  [ocr-warn] {image_path.name}: {e}", file=sys.stderr)
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= min_chars else ""
