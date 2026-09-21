"""Synthetic test images rendered with PIL (no fixtures on disk)."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 640, 360


def _font(mono: bool, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Menlo.ttc", "consola.ttf"]
        if mono
        else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf", "arial.ttf"]
    )
    for name in names:
        for root in ("/usr/share/fonts", "/System/Library/Fonts", "C:/Windows/Fonts"):
            hits = glob.glob(f"{root}/**/{name}", recursive=True)
            if hits:
                return ImageFont.truetype(hits[0], size)
    return ImageFont.load_default()


def _render(bg: str, fg: str, lines: list[str], mono: bool, size: int = 18, x0: int = 30, y0: int = 30) -> Image.Image:
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    font = _font(mono, size)
    y = y0
    for line in lines:
        d.text((x0, y), line, fill=fg, font=font)
        y += int(size * 1.5)
    return img


def slide() -> Image.Image:
    img = _render(
        "white", "black",
        [
            "Why hierarchy matters",
            "",
            "• Size, position and color create contrast",
            "• The eye goes to what is different",
            "• Put the most important thing at the top",
            "• Group related items in a container",
        ],
        mono=False, size=22,
    )
    d = ImageDraw.Draw(img)
    d.text((30, 20), "Why hierarchy matters", fill="black", font=_font(False, 34))
    return img


def terminal() -> Image.Image:
    return _render(
        "black", "#d0d0d0",
        [
            "$ uv sync",
            "Resolved 24 packages in 120ms",
            "Installed 24 packages in 300ms",
            "$ uv run multi --url https://youtu.be/abc",
            "[1/7] downloading video -> output/abc/video.mp4",
            "[2/7] detecting scenes (threshold=27.0)",
            "      ->80 frames",
            "$ ls output/abc/frames | wc -l",
            "80",
            "$ _",
        ],
        mono=True, size=16,
    )


def code() -> Image.Image:
    return _render(
        "#fdf6e3", "#073642",
        [
            "import re",
            "from pathlib import Path",
            "",
            "def parse_transcript(path: Path) -> list[Chunk]:",
            "    raw = path.read_text(encoding=\"utf-8\").strip()",
            "    blocks = re.split(r\"\\n\\s*\\n\", raw)",
            "    chunks: list[Chunk] = []",
            "    for block in blocks:",
            "        lines = block.strip().splitlines()",
            "        if not lines:",
            "            continue",
            "        chunks.append(Chunk(start, text))",
            "    return chunks",
        ],
        mono=True, size=16,
    )


def diagram() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    boxes = [(40, 60, 200, 140), (260, 60, 420, 140), (480, 60, 620, 140), (150, 220, 310, 300), (370, 220, 530, 300)]
    for x0, y0, x1, y1 in boxes:
        d.rectangle((x0, y0, x1, y1), outline="#1f4e79", width=4, fill="#dbe9f6")
    for (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) in zip(boxes, boxes[1:]):
        d.line(((ax0 + ax1) // 2, ay1 if ay1 < by0 else ay0, (bx0 + bx1) // 2, by0 if ay1 < by0 else by1), fill="#1f4e79", width=4)
    font = _font(False, 16)
    for (x0, y0, x1, y1), label in zip(boxes, ["Fetch", "Extract", "Enrich", "Align", "Emit"]):
        d.text((x0 + 40, y0 + 30), label, fill="black", font=font)
    return img


def moved_rectangle(shift: int = 0) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((100 + shift, 100, 180 + shift, 160), fill="red")
    return img


def to_bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def save(img: Image.Image, path: Path, quality: int = 85) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=quality)
    return path
