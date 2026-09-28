# Recreating the README animations

The GIFs are genuine renderer output, captured at fixed angular steps rather than recorded in real time. Each loop contains 96 frames over one full rotation, played at 12.5 frames per second (7.68 seconds per loop). Frames are rendered at 800 × 600, cropped with one fixed rectangle across the entire loop to remove unused black margins, and resized to 480 × 360 for the README, using a shared 256-color palette.

Build the application as described in the [README](../README.md), then run these commands from the repository root:

```powershell
.\build\Release\CSGRayCast.exe cpu helix_complex.txt --animate camera out\helix-frames 96
.\build\Release\CSGRayCast.exe cpu industrial_complex.txt --animate light out\industrial-frames 96
python -m pip install Pillow
python scripts/make_demo_gif.py out/helix-frames docs/renders/helix_orbit.gif
python scripts/make_demo_gif.py out/industrial-frames docs/renders/industrial_lighting.gif
```

The export syntax is:

```text
CSGRayCast.exe <cpu|gpu> <scene_file> --animate <camera|light> <output_directory> <frames>
```

Choose a new output directory for each capture; existing directories are rejected to prevent overwriting previous frames. Frame counts must be between 2 and 1000. Raw BMP frames belong under the ignored `out/` directory, not in Git.

Capture mode uses a hidden SDL window and a camera distance of 10 units to fit the demonstration scenes. `camera` rotates the camera horizontally with a fixed light; `light` rotates the light horizontally with a fixed camera. Both use the same rotation methods as the interactive controls, and rendering uses the normal CPU or GPU path. The last frame stops one angular step before the first so the loop has no duplicated endpoint.

The committed GIFs were captured in CPU mode. GIF playback speed is chosen for presentation and must not be used as an FPS benchmark.
