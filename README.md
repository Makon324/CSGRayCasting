# CSG Ray Casting

CSG Ray Casting is a C++ renderer for constructive solid geometry (CSG), with CPU and CUDA backends. It renders scenes built from spheres, cuboids, cylinders, and cones combined with union, intersection, and difference operations. SDL2 displays the image and handles camera and light controls.

## In motion

![Camera rotating around the helix scene](docs/renders/helix_orbit.gif)

Camera rotation around the helix scene, with a fixed light direction.

![Light rotating around the industrial scene](docs/renders/industrial_lighting.gif)

Light rotation in the industrial scene, with a fixed camera.

These animations were generated with the CPU renderer. Each contains 96 frames and plays at 12.5 frames per second. The playback rate is independent of rendering speed. See [animation capture instructions](docs/animations.md) for the commands used to produce them.

## Example renders

![Rendered CSG operations scene](docs/renders/complex_scene.png)

CSG operations scene ([scene file](complex_scene.txt)).

![Rendered industrial scene](docs/renders/industrial_complex.png)

Industrial scene ([scene file](industrial_complex.txt)).

![Rendered helix scene](docs/renders/helix_complex.png)

Helix scene ([scene file](helix_complex.txt)).

## Requirements

- Windows 10 or 11
- CMake 3.18 or newer
- Visual Studio 2022 with the C++ toolchain
- NVIDIA CUDA Toolkit
- A CUDA-capable NVIDIA GPU — required only for `gpu` mode

CMake first looks for an installed copy of SDL2. If one is not found, it downloads the pinned SDL 2.32.10 release during the first configuration.

## Building

Run the following commands from the repository root:

```powershell
cmake -S CSGRayCast -B build
cmake --build build --config Release
```

With the Visual Studio generator, the executable is created at `build\Release\CSGRayCast.exe`.

## Tests

The unit tests run on the CPU and do not require an active CUDA device:

```powershell
cmake -S CSGRayCast -B build -DBUILD_TESTING=ON
cmake --build build --config Release --target CSGRayCastTests
ctest --test-dir build -C Release --output-on-failure
```

## Benchmarks

The CPU benchmark performs a warm-up and then measures the production parser and tracer across three scenes:

```powershell
cmake -S CSGRayCast -B build -DCSGRAYCAST_BUILD_BENCHMARKS=ON
cmake --build build --config Release --target CSGRayCastBenchmarks
.\build\Release\CSGRayCastBenchmarks.exe . 3 160 120
```

The arguments after the repository path are the iteration count, width, and height, respectively. Results are written to standard output in CSV format.

## Running

The program requires a rendering mode and a path to a scene file:

```text
CSGRayCast.exe <cpu|gpu> <scene_file> [output.bmp]
```

Examples run from the repository root:

```powershell
.\build\Release\CSGRayCast.exe gpu helix_complex.txt
.\build\Release\CSGRayCast.exe cpu industrial_complex.txt
```

The optional third argument renders one frame in a hidden window, saves it as a BMP, and exits:

```powershell
.\build\Release\CSGRayCast.exe cpu complex_scene.txt render.bmp
```

The `cpu` mode does not require an NVIDIA GPU for rendering. The `gpu` mode moves the scene tree and ray calculations to the CUDA device.

## Controls

- **Arrow keys:** Rotate the camera around its target.
- **W/S/A/D:** Rotate the light direction.
- **Close the window:** Exit the program.

## Scene file format

A scene is a single binary CSG tree stored in pre-order. Indentation does not affect parsing, but it makes the structure easier to read. Every non-empty line describes either an operator or a primitive.

### CSG operators

Each operator takes exactly two subtrees, written immediately after it:

- `union` — union of the solids, `A ∪ B`;
- `intersection` — common part of the solids, `A ∩ B`;
- `difference` — difference of the solids, `A \ B`.

### Material

Each primitive line ends with six material values:

```text
r g b diff spec shin
```

- `r g b` — color components in the range from 0 to 1;
- `diff` — diffuse reflection coefficient;
- `spec` — specular reflection coefficient;
- `shin` — shininess exponent.

### Primitives

| Primitive | Syntax | Position meaning |
| --- | --- | --- |
| Sphere | `sphere x y z radius [material]` | Sphere center |
| Cuboid | `cuboid x y z w h d [material]` | Minimum corner |
| Cylinder | `cylinder x y z radius height [material]` | Center of the bottom base |
| Cone | `cone x y z radius height [material]` | Center of the bottom base |

The cylinder and cone are aligned with the Y axis. The `height` value specifies the distance from the bottom base in the positive Y direction.

### Example scene

```text
difference
  sphere 0.0 0.0 0.0 1.4 1.0 0.2 0.2 0.8 0.6 64
  cuboid -1.1 -1.1 -1.1 2.2 2.2 2.2 0.2 0.2 1.0 0.8 0.5 32
```

This definition subtracts the cuboid from the sphere.

## Scene generator

The `gen_scene.py` script creates procedural city scenes with an approximate node count:

```powershell
python gen_scene.py 500 generated_city.txt
```

The repository includes single-solid scenes, examples of CSG operations, and city scenes in `big_city.txt` and `large_city.txt`.

## Implementation

### Intersections and CSG operations

Each primitive returns a `Span` interval along the ray that lies inside the solid. The interval contains entry and exit parameters (`t_entry` and `t_exit`), surface normals, and a material identifier.

Operators combine sorted intervals:

- **Union:** Merges overlapping intervals.
- **Intersection:** Keeps only their common portion.
- **Difference:** Removes portions belonging to the right-hand object from the left-hand object's intervals.

The closest positive intersection after evaluating the complete tree determines the pixel color.

### Flat tree representation

`FlatCSGTree` stores the tree in arrays instead of a pointer-based structure. Its topology is held in the `nodes`, `left_indexes`, and `right_indexes` arrays, while primitive and material data are compacted separately.

The renderer evaluates the tree iteratively in post-order using a stack. The CPU and GPU backends use the same tree representation.

### CPU and GPU memory

The CPU renderer allocates scratch buffers once per frame and reuses them for subsequent rays.

The GPU renderer calculates the required pool size before launching the kernel. A global buffer is divided among pixels, and rendering proceeds in batches constrained by the memory budget. Tree topology and primitive data are copied into shared memory for each thread block.

## Project structure

- `CSGRayCast/main.cu` — entry point, SDL handling, and CPU/GPU rendering.
- `CSGRayCast/tracer.cu` — ray tracing, interval operations, and the CUDA kernel.
- `CSGRayCast/shape.h` — analytical intersections for spheres, cuboids, cylinders, and cones.
- `CSGRayCast/csg.h` — flat CSG tree representation.
- `CSGRayCast/loadfile.cpp` — scene file parser.
- `CSGRayCast/rayCast.h` — vectors, rays, camera, light, and colors.
- `gen_scene.py` — procedural city scene generator.

## License

This project is licensed under the [MIT License](LICENSE).
