"""Best-effort reconstruction of code files and shell commands from code/terminal frames."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Scene, hms

_EXTS = (
    "py|js|ts|tsx|jsx|mjs|cjs|go|rs|rb|java|kt|swift|c|h|cpp|hpp|cs|php|sh|bash|zsh|ps1|"
    "json|yaml|yml|toml|ini|cfg|md|sql|css|scss|html|vue|svelte|txt|env|dockerfile"
)
_FILENAME_RE = re.compile(rf"(?<![\w/.-])((?:[\w\-]+/)*[\w\-]+\.(?:{_EXTS}))(?![\w.-])", re.I)
_LANG_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("py", re.compile(r"^\s*(def |class |import |from \w+ import |elif |print\(|self\.)|:\s*$")),
    ("ts", re.compile(r"\b(interface |type \w+ =|: (string|number|boolean)\b|import .* from ['\"]|export (const|default|function))")),
    ("js", re.compile(r"\b(const |let |var |function |=>|console\.log|require\()")),
    ("go", re.compile(r"\b(func |package |:=|fmt\.)")),
    ("rs", re.compile(r"\b(fn |let mut |impl |pub |use \w+::)")),
    ("rb", re.compile(r"^\s*(def |end$|require |puts )|\bdo \|")),
    ("java", re.compile(r"\b(public (class|static|void)|System\.out|private \w+ \w+;)")),
    ("sh", re.compile(r"^\s*(#!/bin/|export \w+=|echo |if \[|fi$|done$)")),
    ("html", re.compile(r"</?(div|span|html|body|head|p|a|ul|li|script|style)\b")),
    ("css", re.compile(r"^\s*[.#]?[\w-]+\s*\{|^\s*[\w-]+\s*:\s*[^;]+;")),
    ("sql", re.compile(r"\b(SELECT|INSERT INTO|CREATE TABLE|FROM|WHERE)\b")),
    ("json", re.compile(r"^\s*\"[\w-]+\"\s*:\s")),
    ("yaml", re.compile(r"^\s*[\w-]+:\s*(\S|$)|^\s*- ")),
]
_COMMENT = {
    "py": "#", "rb": "#", "sh": "#", "bash": "#", "zsh": "#", "yaml": "#", "yml": "#", "toml": "#", "ini": "#",
    "cfg": "#", "ps1": "#", "dockerfile": "#", "env": "#",
    "js": "//", "ts": "//", "tsx": "//", "jsx": "//", "mjs": "//", "cjs": "//", "go": "//", "rs": "//", "java": "//",
    "kt": "//", "swift": "//", "c": "//", "h": "//", "cpp": "//", "hpp": "//", "cs": "//", "php": "//", "scss": "//",
    "sql": "--", "html": "<!--", "vue": "<!--", "svelte": "<!--", "md": "<!--", "css": "/*", "txt": "#",
}
_PROMPT_RE = re.compile(
    r"^\s*(?:\(\w+\)\s*)?(?:(?:[\w~./-]*[@:]){0,2}[\w~./-]*\s*)?(?:PS [A-Z]:\\[^>]*>|\$|>|❯|»|%|#|~\$)\s+(.+?)\s*$"
)
_LOW_CONFIDENCE = 60.0


@dataclass
class Reconstructed:
    path: Path
    kind: str  # "code" | "terminal"
    frame_idxs: list[int]
    confidence: float  # 0..1, mean OCR word confidence of contributing frames
    uncertain_lines: list[int] = field(default_factory=list)  # 1-based line numbers flagged in the file
    filename_source: str = "guessed"  # "ocr" | "guessed"
    n_lines: int = 0


# --- guessing ---------------------------------------------------------------------------


def guess_filename(ocr_lines: list[str]) -> str | None:
    """A filename seen in the OCR (editor tab / title bar first, then anywhere)."""
    for window in (ocr_lines[:4], ocr_lines):
        for line in window:
            m = _FILENAME_RE.search(line)
            if m:
                name = m.group(1)
                if not name.lower().startswith(("http", "www.")):
                    return name.replace("\\", "/").split("/")[-1]
    return None


def guess_extension(ocr_lines: list[str]) -> str:
    scores: dict[str, int] = {}
    for ext, pat in _LANG_HINTS:
        scores[ext] = sum(1 for line in ocr_lines if pat.search(line))
    best = max(scores.items(), key=lambda kv: kv[1], default=("txt", 0))
    return best[0] if best[1] >= 2 else "txt"


def comment_prefix(ext: str) -> str | None:
    if ext in ("json",):
        return None
    return _COMMENT.get(ext, "#")


# --- folding snapshots -------------------------------------------------------------------


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def fold_snapshots(snapshots: list[list[str]]) -> list[tuple[str, int]]:
    """Fold successive OCR snapshots of an evolving file into one buffer.

    Returns [(line, times_seen)]. Later snapshots win on replaced lines. Lines
    that vanish from the *middle* of the buffer are treated as deleted; lines
    that vanish at the top or bottom are treated as scrolled out and kept.
    """
    buffer: list[str] = []
    seen: list[int] = []
    for snap in snapshots:
        snap = [_norm(l) for l in snap if _norm(l)]
        if not buffer:
            buffer, seen = list(snap), [1] * len(snap)
            continue
        sm = difflib.SequenceMatcher(None, buffer, snap, autojunk=False)
        new_buf: list[str] = []
        new_seen: list[int] = []
        opcodes = sm.get_opcodes()
        for k, (tag, i1, i2, j1, j2) in enumerate(opcodes):
            if tag == "equal":
                new_buf.extend(buffer[i1:i2])
                new_seen.extend(c + 1 for c in seen[i1:i2])
            elif tag == "insert":
                new_buf.extend(snap[j1:j2])
                new_seen.extend([1] * (j2 - j1))
            elif tag == "replace":
                old, new = buffer[i1:i2], snap[j1:j2]
                # Pair up near-identical lines (OCR noise) so their counts carry over.
                for jj, line in enumerate(new):
                    match = difflib.get_close_matches(line, old, n=1, cutoff=0.8)
                    new_buf.append(line)
                    new_seen.append(seen[i1 + old.index(match[0])] + 1 if match else 1)
                    _ = jj
            elif tag == "delete":
                at_edge = k == 0 or k == len(opcodes) - 1
                if at_edge:  # scrolled out of view: keep
                    new_buf.extend(buffer[i1:i2])
                    new_seen.extend(seen[i1:i2])
        buffer, seen = new_buf, new_seen
    return list(zip(buffer, seen))


# --- runs of frames ---------------------------------------------------------------------


def _runs(scenes: list[Scene]) -> list[tuple[str, list[Scene]]]:
    """Consecutive kept code frames split by filename changes; all terminal frames in one run."""
    runs: list[tuple[str, list[Scene]]] = []
    current_name: str | None = None
    current: list[Scene] = []
    unknown = 0
    for s in scenes:
        if not s.kept or s.kind != "code" or not s.ocr_lines:
            if current:
                runs.append((current_name or "", current))
                current, current_name = [], None
            continue
        name = guess_filename(s.ocr_lines)
        if current and name and current_name and name != current_name:
            runs.append((current_name, current))
            current, current_name = [], None
        if not current:
            current_name = name
        elif name and not current_name:
            current_name = name
        current.append(s)
    if current:
        runs.append((current_name or "", current))
    out: list[tuple[str, list[Scene]]] = []
    for name, frames in runs:
        if not name:
            unknown += 1
            name = f"unknown_{unknown}.{guess_extension([l for f in frames for l in f.ocr_lines])}"
        out.append((name, frames))
    return out


def _unique_path(directory: Path, name: str, used: set[str]) -> Path:
    base = name
    n = 2
    while name in used:
        stem, dot, ext = base.rpartition(".")
        name = f"{stem}_{n}.{ext}" if dot else f"{base}_{n}"
        n += 1
    used.add(name)
    return directory / name


def reconstruct_code(scenes: list[Scene], out_dir: Path) -> list[Reconstructed]:
    results: list[Reconstructed] = []
    used: set[str] = set()
    for name, frames in _runs(scenes):
        folded = fold_snapshots([f.ocr_lines for f in frames])
        if not folded:
            continue
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
        prefix = comment_prefix(ext)
        conf = sum(f.ocr_confidence for f in frames) / len(frames) / 100.0
        n_snaps = len(frames)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = _unique_path(out_dir, name, used)
        uncertain: list[int] = []
        lines: list[str] = []
        header_len = 0
        if prefix is not None:
            close = " -->" if prefix == "<!--" else " */" if prefix == "/*" else ""
            header = [
                f"{prefix} reconstructed by multi from {n_snaps} frame(s): {', '.join(f'{f.idx:04d}@{hms(f.t_seconds)}' for f in frames)}{close}",
                f"{prefix} mean OCR confidence {conf:.2f}; lines marked '?? ' below appeared in only one frame or scored low{close}",
                "",
            ]
            lines.extend(header)
            header_len = len(header)
        for i, (line, times) in enumerate(folded, start=1):
            flagged = (n_snaps >= 3 and times == 1) or conf * 100 < _LOW_CONFIDENCE
            if flagged:
                uncertain.append(header_len + i)
                if prefix is not None:
                    close = " -->" if prefix == "<!--" else " */" if prefix == "/*" else ""
                    line = f"{line}  {prefix} ??{close}"
            lines.append(line)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        results.append(Reconstructed(
            path=path, kind="code", frame_idxs=[f.idx for f in frames], confidence=round(conf, 2),
            uncertain_lines=uncertain, filename_source="ocr" if not name.startswith("unknown_") else "guessed",
            n_lines=len(folded),
        ))
    return results


def extract_commands(ocr_lines: list[str]) -> list[str]:
    cmds: list[str] = []
    for line in ocr_lines:
        m = _PROMPT_RE.match(line)
        if m:
            cmd = m.group(1).strip()
            if len(cmd) >= 2 and not cmd.startswith(("_", "|")):
                cmds.append(cmd)
    return cmds


def reconstruct_commands(scenes: list[Scene], out_dir: Path) -> Reconstructed | None:
    entries: list[tuple[Scene, str]] = []
    last: str | None = None
    for s in scenes:
        if not s.kept or s.kind != "terminal":
            continue
        for cmd in extract_commands(s.ocr_lines):
            if cmd != last:
                entries.append((s, cmd))
                last = cmd
    if not entries:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "commands.sh"
    frames = sorted({s.idx for s, _ in entries})
    lines = ["#!/usr/bin/env bash", "# Commands seen in terminal frames, reconstructed by multi (best effort; verify before running).", ""]
    uncertain: list[int] = []
    for s, cmd in entries:
        lines.append(f"# t={hms(s.t_seconds)} frame {s.idx:04d} confidence={s.ocr_confidence / 100:.2f}")
        if s.ocr_confidence < _LOW_CONFIDENCE:
            uncertain.append(len(lines) + 1)
            lines.append(f"# ?? {cmd}")
        else:
            lines.append(cmd)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    conf = sum(s.ocr_confidence for s, _ in entries) / len(entries) / 100.0
    return Reconstructed(path=path, kind="terminal", frame_idxs=frames, confidence=round(conf, 2),
                         uncertain_lines=uncertain, filename_source="fixed", n_lines=len(entries))


def reconstruct_all(scenes: list[Scene], out_dir: Path) -> list[Reconstructed]:
    results = reconstruct_code(scenes, out_dir)
    cmds = reconstruct_commands(scenes, out_dir)
    if cmds is not None:
        results.append(cmds)
    if results:
        readme = ["# reconstructed/", "", "Best-effort text reconstructed from OCR of code/terminal frames. Verify against the frames before relying on it.", "",
                  "| File | Kind | Frames | Lines | OCR confidence | Uncertain lines |", "|---|---|---|---|---|---|"]
        for r in results:
            readme.append(f"| `{r.path.name}` | {r.kind} | {', '.join(f'{i:04d}' for i in r.frame_idxs)} | {r.n_lines} | {r.confidence:.2f} | "
                          f"{', '.join(map(str, r.uncertain_lines)) or '–'} |")
        (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return results
