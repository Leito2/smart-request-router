"""Assemble docs/images/surge.gif from the Grafana R3 frames captured during `make surge-demo --hold 70`
(docs/images/surge-frames/*.jpg, in name order). Needs Pillow.

    uv run --no-sync python scripts/make_surge_gif.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS = ["1. steady traffic: ~10 msg/s, EWMA baseline per route",
            "2. `payments` x5 starts: the blue line leaves the pack",
            "3. the detector fires (z-score above threshold for 2 windows)",
            "4. alerts hold; the rules model smears part of the burst onto other routes"]


def main() -> None:
    try:
        font = ImageFont.truetype("consola.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    frames = []
    for path, caption in zip(sorted((ROOT / "docs" / "images" / "surge-frames").glob("*.jpg")), CAPTIONS,
                             strict=True):
        image = Image.open(path).convert("RGB")
        canvas = Image.new("RGB", (image.width, image.height + 28), "#0d1117")
        canvas.paste(image, (0, 0))
        ImageDraw.Draw(canvas).text((10, image.height + 5), caption, font=font, fill="#e6edf3")
        frames.append(canvas)
    out = ROOT / "docs" / "images" / "surge.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[2200, 2200, 2600, 4500], loop=0)
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
