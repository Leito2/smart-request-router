# ruff: noqa: E501, E741
"""Render docs/results/demo.json (written by scripts/demo.py) as a terminal-style GIF: docs/images/demo.gif.

    uv run --no-sync python scripts/make_demo_gif.py

The text is exactly what `make demo` printed for that run; only the typing animation is synthetic.
Needs Pillow and a monospace font (Consolas on Windows, DejaVu Sans Mono elsewhere).
"""
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
WIDTH_CHARS, FONT_SIZE, PAD = 104, 15, 14
BG, FG, DIM, GREEN, YELLOW, CYAN = "#0d1117", "#e6edf3", "#8b949e", "#3fb950", "#d29922", "#58a6ff"


def font() -> ImageFont.FreeTypeFont:
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    raise SystemExit("no monospace font found")


def lines_for(case: dict, number: int) -> list[tuple[str, str]]:
    out = [("", FG), (f"[{number}] {case['case']}", CYAN)]
    out += [(l, FG) for l in textwrap.wrap(f'"{case["text"]}"', WIDTH_CHARS - 4, initial_indent="    ",
                                           subsequent_indent="    ")]
    if "escalated_because" not in case:
        out.append((f"    System 1 ({case['system1_ms']:.0f} ms): route = {case['route']}, priority = "
                    f"{case['priority']}, confidence = {case['confidence']}", GREEN))
        return out
    out.append((f"    System 1 ({case['system1_ms']:.0f} ms): escalated, reason = {case['escalated_because']}",
                YELLOW))
    out.append((f"    -> {case['decided_by']}: route = {case['route']}, priority = {case.get('priority', '-')}", GREEN))
    if case.get("rationale"):
        out += [(l, DIM) for l in textwrap.wrap(f"why: {case['rationale']}", WIDTH_CHARS - 7,
                                                initial_indent="       ", subsequent_indent="            ")]
    if case.get("tools_called"):
        out.append((f"       tools: {', '.join(case['tools_called'])}", DIM))
    if case.get("served_by"):
        out.append((f"       LLM: {', '.join(case['served_by'])} in {case['system2_s']} s", DIM))
    return out


def main() -> None:
    cases = json.loads((ROOT / "docs" / "results" / "demo.json").read_text(encoding="utf-8").replace("‑", "-"))
    script = [("$ make demo", FG)] + [line for n, c in enumerate(cases, 1) for line in lines_for(c, n)]
    f = font()
    line_h = FONT_SIZE + 5
    height_lines = 26                                  # the window scrolls like a terminal
    width = int(f.getlength("M") * WIDTH_CHARS) + 2 * PAD
    height = line_h * height_lines + 2 * PAD
    frames, durations = [], []
    for end in range(1, len(script) + 1):
        shown = script[max(0, end - height_lines):end]
        image = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(image)
        for row, (text, color) in enumerate(shown):
            draw.text((PAD, PAD + row * line_h), text, font=f, fill=color)
        frames.append(image)
        slow = shown[-1][0].lstrip().startswith(("[", "->"))
        durations.append(900 if slow else 350)
    durations[-1] = 4000
    out = ROOT / "docs" / "images" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
