from __future__ import annotations

from pathlib import Path

from youtube_multi.models import Scene
from youtube_multi.reconstruct import (
    extract_commands,
    fold_snapshots,
    guess_extension,
    guess_filename,
    reconstruct_all,
    reconstruct_code,
    reconstruct_commands,
)


def test_guess_filename_prefers_title_bar() -> None:
    assert guess_filename(["main.py — Visual Studio Code", "import os", "utils.py"]) == "main.py"
    assert guess_filename(["src/app/routes.ts", "export const x = 1"]) == "routes.ts"
    assert guess_filename(["visit https://example.com/index.html today"]) is None  # URLs are not editor tabs
    assert guess_filename(["no files here", "just prose"]) is None
    assert guess_filename([]) is None


def test_guess_extension() -> None:
    assert guess_extension(["def main():", "    return 0", "import sys"]) == "py"
    assert guess_extension(["const a = 1;", "let b = () => 2;", "console.log(a)"]) == "js"
    assert guess_extension(["func main() {", "  x := 1", "  fmt.Println(x)"]) == "go"
    assert guess_extension(["hello", "world"]) == "txt"


def test_fold_snapshots_latest_wins_and_scroll_is_kept() -> None:
    s1 = ["import os", "", "def main():", "    return 0"]
    s2 = ["import os", "def main():", "    print('hi')", "    return 0"]  # inserted a line
    s3 = ["def main():", "    print('hi')", "    return 0", "", "main()"]  # scrolled down: 'import os' left the top
    folded = fold_snapshots([s1, s2, s3])
    lines = [l for l, _ in folded]
    assert lines == ["import os", "def main():", "print('hi')", "return 0", "main()"]
    counts = dict(folded)
    assert counts["def main():"] == 3 and counts["main()"] == 1


def test_fold_snapshots_middle_deletion_and_ocr_noise() -> None:
    s1 = ["a = 1", "b = 2", "c = 3"]
    s2 = ["a = 1", "c = 3"]  # b deleted from the middle
    s3 = ["a = 1", "c = 3", "d = 4"]  # OCR noise: '3' read as '3'
    folded = fold_snapshots([s1, s2, s3])
    assert [l for l, _ in folded] == ["a = 1", "c = 3", "d = 4"]
    assert dict(folded)["a = 1"] == 3  # counts carried through the near-identical replace
    assert fold_snapshots([]) == []
    assert fold_snapshots([[], ["x"]]) == [("x", 1)]


def test_extract_commands() -> None:
    lines = ["$ uv sync", "Resolved 3 packages", "user@host:~/proj$ git status", "PS C:\\Users\\me> npm install", "❯ ls -la", "> _", "$ "]
    assert extract_commands(lines) == ["uv sync", "git status", "npm install", "ls -la"]


def _frame(idx: int, t: float, kind: str, lines: list[str], conf: float = 90.0) -> Scene:
    s = Scene(idx=idx, t_seconds=t, image_path=Path(f"frames/{idx:04d}.jpg"))
    s.kind, s.ocr_lines, s.ocr_confidence = kind, lines, conf
    return s


def test_reconstruct_code_writes_named_file_with_header_and_flags(tmp_path: Path) -> None:
    frames = [
        _frame(1, 10, "code", ["app.py", "import os", "def main():", "    pass"]),
        _frame(2, 20, "code", ["app.py", "import os", "def main():", "    print('x')"]),
        _frame(3, 30, "code", ["app.py", "import os", "def main():", "    print('x')", "main()"]),
        _frame(4, 40, "slide", ["Not code"]),
        _frame(5, 50, "code", ["const a = 1;", "let b = 2;", "console.log(a + b);"]),
    ]
    results = reconstruct_code(frames, tmp_path / "reconstructed")
    assert [r.path.name for r in results] == ["app.py", "unknown_1.js"]
    app = results[0]
    text = app.path.read_text()
    assert text.startswith("# reconstructed by multi from 3 frame(s): 0001@00:00:10, 0002@00:00:20, 0003@00:00:30")
    assert "def main():" in text and "print('x')" in text
    assert "main()  # ??" in text  # seen in only one of three frames -> flagged
    assert "pass" not in text  # replaced by the later frame
    assert app.frame_idxs == [1, 2, 3] and app.confidence == 0.9 and app.filename_source == "ocr"
    assert app.uncertain_lines and all(text.splitlines()[n - 1].endswith("# ??") for n in app.uncertain_lines)
    js = results[1]
    assert js.filename_source == "guessed" and js.frame_idxs == [5]
    assert "// reconstructed by multi" in js.path.read_text()


def test_reconstruct_code_splits_on_filename_change_and_dedupes_names(tmp_path: Path) -> None:
    frames = [
        _frame(1, 0, "code", ["a.py", "x = 1"]),
        _frame(2, 5, "code", ["b.py", "y = 2"]),
        _frame(3, 9, "terminal", ["$ ls"]),
        _frame(4, 12, "code", ["a.py", "x = 3"]),
    ]
    results = reconstruct_code(frames, tmp_path)
    assert [r.path.name for r in results] == ["a.py", "b.py", "a_2.py"]


def test_reconstruct_commands(tmp_path: Path) -> None:
    frames = [
        _frame(1, 0, "terminal", ["$ uv sync", "ok"]),
        _frame(2, 5, "terminal", ["$ uv sync", "ok", "$ uv run multi --url x"]),  # repeated first command is not duplicated
        _frame(3, 9, "terminal", ["$ rm -rf build"], conf=40.0),  # low confidence -> commented out
        _frame(4, 12, "code", ["$ not a terminal"]),
    ]
    r = reconstruct_commands(frames, tmp_path)
    assert r is not None and r.path.name == "commands.sh"
    text = r.path.read_text()
    assert text.count("uv sync") == 1
    assert "# t=00:00:05 frame 0002 confidence=0.90\nuv run multi --url x" in text
    assert "# ?? rm -rf build" in text
    assert r.frame_idxs == [1, 2, 3] and r.uncertain_lines
    assert reconstruct_commands([frames[3]], tmp_path) is None


def test_reconstruct_all_writes_readme(tmp_path: Path) -> None:
    frames = [_frame(1, 0, "code", ["x.py", "a = 1"]), _frame(2, 3, "terminal", ["$ echo hi"])]
    results = reconstruct_all(frames, tmp_path / "r")
    assert {r.path.name for r in results} == {"x.py", "commands.sh"}
    readme = (tmp_path / "r" / "README.md").read_text()
    assert "| `x.py` | code | 0001 |" in readme and "commands.sh" in readme
    assert reconstruct_all([], tmp_path / "empty") == [] and not (tmp_path / "empty").exists()
