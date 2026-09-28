"""Encode an exported renderer frame sequence as a small, looping README GIF.

Requires Pillow: python -m pip install Pillow
"""

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="Directory containing frame_0000.bmp, etc.")
    parser.add_argument("output", type=Path, help="Destination .gif file")
    args = parser.parse_args()
    paths = sorted(args.frames.glob("frame_*.bmp"))
    if len(paths) < 2:
        parser.error("At least two exported frames are required.")
    if [path.name for path in paths] != [f"frame_{i:04d}.bmp" for i in range(len(paths))]:
        parser.error("Frames must be consecutively numbered starting at frame_0000.bmp.")
    if args.output.suffix.lower() != ".gif":
        parser.error("Output must have a .gif extension.")

    frames = []
    for path in paths:
        with Image.open(path) as source:
            if source.size != (800, 600):
                parser.error(f"Expected 800x600 renderer output: {path}")
            frames.append(source.convert("RGB"))

    # Apply one fixed crop to the entire sequence, removing unused black margins
    # without introducing per-frame zoom or changing the renderer's motion.
    bounds = [box for frame in frames if (box := frame.getbbox()) is not None]
    if not bounds:
        parser.error("All frames are empty.")
    left = min(box[0] for box in bounds)
    top = min(box[1] for box in bounds)
    right = max(box[2] for box in bounds)
    bottom = max(box[3] for box in bounds)
    width = max(right - left + 32, (bottom - top + 32) * 4 / 3)
    height = width * 3 / 4
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    crop = tuple(round(value) for value in (
        center_x - width / 2, center_y - height / 2,
        center_x + width / 2, center_y + height / 2,
    ))
    frames = [frame.crop(crop).resize((480, 360), Image.Resampling.LANCZOS) for frame in frames]

    # One palette sampled across the whole loop avoids frame-to-frame color flicker.
    samples = Image.new("RGB", (80 * len(frames), 60))
    for index, frame in enumerate(frames):
        samples.paste(frame.resize((80, 60), Image.Resampling.LANCZOS), (80 * index, 0))
    palette = samples.quantize(colors=256)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        args.output,
        save_all=True,
        append_images=indexed[1:],
        duration=80,
        loop=0,
        disposal=2,
        optimize=True,
    )
    with Image.open(args.output) as result:
        if result.n_frames != len(frames) or result.info.get("loop") != 0:
            raise RuntimeError("Encoded GIF failed frame-count or looping validation.")
    print(f"Saved {args.output}: {len(frames)} frames, {args.output.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
