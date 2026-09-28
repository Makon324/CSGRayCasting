#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <SDL.h>
#include <chrono>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

#include "csg.h"
#include "rayCast.h"
#include "tracer.h"

// --- CONSTANTS ---
constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;
constexpr float ROTATION_SPEED = 1.0f;  // Radians per second
constexpr size_t MAX_SCRATCH_MEMORY_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr int threadsPerBlock = 256;

void cpuRender(Color* h_image, const Camera& cam, const Light& light, const FlatCSGTree& tree) {
    // We allocate enough memory for the worst-case CSG operation defined by the tree.
    // This memory persists for the entire frame render.
    std::vector<Span> pool_buffer(tree.max_pool_size);
    std::vector<StackEntry> stack_buffer(tree.max_stack_depth);

    // Create the wrappers that the tracer expects
    StridedSpan pool(pool_buffer.data(), 1);
    StridedStack stack(stack_buffer.data(), 1);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float s = (x + 0.5f) / WIDTH;
            float t = (y + 0.5f) / HEIGHT;
            Ray ray = cam.getRay(s, t);

            h_image[y * WIDTH + x] = trace(ray, light, tree, pool, stack);
        }
    }
}

void gpuRender(Color* h_image, Color* d_image, const Camera& cam, const Light& light, const FlatCSGTree& d_tree,
    Span* d_global_pool, StackEntry* d_global_stack, size_t batch_size) {

    // Shared memory size calculation
    size_t shared_size =
        // Topology (per Node)
        d_tree.num_nodes * (
            sizeof(FlatCSGNodeInfo) +
            3 * sizeof(uint32_t) +       // left, right, post_order
            sizeof(int32_t)              // primitive_idx
        ) +
        // Data (per Primitive)
        d_tree.num_primitives * (
            MAX_SHAPE_DATA_SIZE * sizeof(float) + // Shape data
            6 * sizeof(float)                     // Material props (rgb + 3 coeffs)
        );

    // Align shared_size to be safe (optional but recommended)
    if (shared_size % 8 != 0) shared_size += (8 - (shared_size % 8));

    size_t total_pixels = static_cast<size_t>(WIDTH) * HEIGHT;

    // We iterate through pixels in chunks of 'batch_size'
    for (size_t offset = 0; offset < total_pixels; offset += batch_size) {

        // Calculate the actual size of the current batch (last batch might be smaller)
        size_t current_batch_count = std::min(batch_size, total_pixels - offset);

        // Calculate grid size for this batch
        int blocksPerGrid = static_cast<int>((current_batch_count + threadsPerBlock - 1) / threadsPerBlock);

        // Launch kernel passing the offset
        renderKernel << <blocksPerGrid, threadsPerBlock, shared_size >> > (
            d_image, cam, light, d_tree,
            d_global_pool, d_global_stack,
            offset, current_batch_count, total_pixels
        );

        checkCudaError(cudaGetLastError(), "renderKernel launch");
    }

    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    checkCudaError(cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * sizeof(Color), cudaMemcpyDeviceToHost), "cudaMemcpy to host");
}

void compactTreeMemory(FlatCSGTree& tree) {
    size_t num_nodes = tree.num_nodes;
    size_t prim_count = 0;

    // Calculate Primitive Count and Assign Indices
    for (size_t i = 0; i < num_nodes; ++i) {
        if (tree.nodes[i].shape_type != ShapeType::TreeNode) {
            tree.primitive_idx[i] = (int32_t)prim_count++;
        }
        else {
            tree.primitive_idx[i] = -1;
        }
    }
    tree.num_primitives = prim_count;

    std::cout << "Compacting Tree: " << num_nodes << " nodes -> " << prim_count << " primitives." << std::endl;

    // Allocate New Compact Buffers
    float* new_data = new float[prim_count * MAX_SHAPE_DATA_SIZE];
    float* new_red = new float[prim_count];
    float* new_green = new float[prim_count];
    float* new_blue = new float[prim_count];
    float* new_diff = new float[prim_count];
    float* new_spec = new float[prim_count];
    float* new_shin = new float[prim_count];

    // Move Data
    for (size_t i = 0; i < num_nodes; ++i) {
        int32_t p_idx = tree.primitive_idx[i];
        if (p_idx != -1) {
            // Copy Shape Data
            std::memcpy(&new_data[p_idx * MAX_SHAPE_DATA_SIZE],
                &tree.data[i * MAX_SHAPE_DATA_SIZE],
                MAX_SHAPE_DATA_SIZE * sizeof(float));

            // Copy Materials
            new_red[p_idx] = tree.red[i];
            new_green[p_idx] = tree.green[i];
            new_blue[p_idx] = tree.blue[i];
            new_diff[p_idx] = tree.diffuse_coeff[i];
            new_spec[p_idx] = tree.specular_coeff[i];
            new_shin[p_idx] = tree.shininess[i];
        }
    }

    // Swap and Delete Old Buffers
    delete[] tree.data; tree.data = new_data;
    delete[] tree.red; tree.red = new_red;
    delete[] tree.green; tree.green = new_green;
    delete[] tree.blue; tree.blue = new_blue;
    delete[] tree.diffuse_coeff; tree.diffuse_coeff = new_diff;
    delete[] tree.specular_coeff; tree.specular_coeff = new_spec;
    delete[] tree.shininess; tree.shininess = new_shin;
}

void updateSurface(SDL_Surface* surface, Color* h_image) {
    Uint8* pixels = static_cast<Uint8*>(surface->pixels);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            Color c = h_image[y * WIDTH + x];
            Uint8 r = static_cast<Uint8>(std::min(1.f, std::max(0.f, c.r)) * 255);
            Uint8 g = static_cast<Uint8>(std::min(1.f, std::max(0.f, c.g)) * 255);
            Uint8 b = static_cast<Uint8>(std::min(1.f, std::max(0.f, c.b)) * 255);
            Uint32* pixel = reinterpret_cast<Uint32*>(pixels + y * surface->pitch + x * 4);
            *pixel = SDL_MapRGB(surface->format, r, g, b);
        }
    }
}

int main(int argc, char** argv) {
    const bool animate = argc >= 4 && std::strcmp(argv[3], "--animate") == 0;
    if (argc < 3 || (std::strcmp(argv[1], "cpu") != 0 && std::strcmp(argv[1], "gpu") != 0)
        || (animate ? argc != 7 : argc > 4)) {
        std::cerr << "Usage: " << argv[0] << " <cpu|gpu> <scene_file> [output.bmp]\n"
            << "       " << argv[0] << " <cpu|gpu> <scene_file> --animate <camera|light> <output_directory> <frames>\n";
        return 1;
    }
    bool use_gpu = (std::strcmp(argv[1], "gpu") == 0);
    const char* output_file = !animate && argc == 4 ? argv[3] : nullptr;
    int frame_count = 0;
    bool animate_camera = false;
    if (animate) {
        try {
            animate_camera = std::strcmp(argv[4], "camera") == 0;
            if (!animate_camera && std::strcmp(argv[4], "light") != 0)
                throw std::runtime_error("Animation must be camera or light.");
            size_t consumed = 0;
            frame_count = std::stoi(argv[6], &consumed);
            if (consumed != std::strlen(argv[6]) || frame_count < 2 || frame_count > 1000)
                throw std::runtime_error("Frame count must be an integer from 2 to 1000.");
            // Require a new directory so an export cannot overwrite earlier captures.
            if (!std::filesystem::create_directories(argv[5]))
                throw std::runtime_error("Output directory already exists; choose a new directory.");
        }
        catch (const std::exception& e) {
            std::cerr << "[Error] " << e.what() << std::endl;
            return 1;
        }
    }

    FlatCSGTree h_tree;
    try {
        h_tree = loadFromFile(argv[2]);
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] Failed to load scene file: " << e.what() << std::endl;
        return 1;
    }

    // Check for empty tree to prevent Division By Zero later
    if (h_tree.num_nodes == 0) {
        std::cerr << "[Error] The scene file is empty or contains no valid nodes." << std::endl;
        return 1;
    }

    compactTreeMemory(h_tree);

    // Compute sizes based on tree
    h_tree.max_pool_size = computeTotalSpanUsage(h_tree);
    h_tree.max_stack_depth = computeMaxDepth(h_tree) * 2;

    Vec3 lookat(0, 0, 0);
    Vec3 up(0, 1, 0);
    float fov = 60.0f;
    Light light(Vec3(1, 1, 1));
    Color* h_image = new Color[WIDTH * HEIGHT];
    Color* d_image = nullptr;

    // Global Memory Buffers for GPU
    Span* d_global_pool = nullptr;
    StackEntry* d_global_stack = nullptr;
    size_t batch_size = 0;

    FlatCSGTree d_tree;
    if (use_gpu) {
        checkCudaError(cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(Color)), "cudaMalloc d_image");
        copyTreeToDevice(h_tree, d_tree);

        // SMART MEMORY ALLOCATION
        size_t total_pixels = static_cast<size_t>(WIDTH) * HEIGHT;

        // Calculate memory required per pixel
        size_t bytes_per_pixel = (static_cast<size_t>(h_tree.max_pool_size) * sizeof(Span)) +
            (static_cast<size_t>(h_tree.max_stack_depth) * sizeof(StackEntry));

        // Calculate how many pixels fit in our memory budget
        batch_size = MAX_SCRATCH_MEMORY_BYTES / bytes_per_pixel;

        // Safety clamps
        if (batch_size == 0) batch_size = 1;
        if (batch_size > total_pixels) batch_size = total_pixels;

        size_t pool_alloc_size = batch_size * h_tree.max_pool_size * sizeof(Span);
        size_t stack_alloc_size = batch_size * h_tree.max_stack_depth * sizeof(StackEntry);

        std::cout << "Initialization:\n"
            << "  Resolution: " << WIDTH << "x" << HEIGHT << "\n"
            << "  Per Pixel Reqs: " << bytes_per_pixel / 1024.0 << " KB\n"
            << "  Memory Limit: " << MAX_SCRATCH_MEMORY_BYTES / (1024.0 * 1024.0) << " MB\n"
            << "  Batch Size: " << batch_size << " pixels (out of " << total_pixels << ")\n"
            << "  Allocating Pool: " << pool_alloc_size / (1024.0 * 1024.0) << " MB\n"
            << "  Allocating Stack: " << stack_alloc_size / (1024.0 * 1024.0) << " MB\n";

        checkCudaError(cudaMalloc(&d_global_pool, pool_alloc_size), "cudaMalloc global pool");
        checkCudaError(cudaMalloc(&d_global_stack, stack_alloc_size), "cudaMalloc global stack");
    }

    SDL_Init(SDL_INIT_VIDEO);
    Uint32 window_flags = (output_file || animate) ? SDL_WINDOW_HIDDEN : 0;
    SDL_Window* window = SDL_CreateWindow("CSG Ray Tracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, window_flags);
    SDL_Surface* surface = SDL_GetWindowSurface(window);
    float angle = 0.0f;
    Vec3 initial_origin(5.0f * sinf(angle), 0.0f, 5.0f * cosf(angle));
    Camera cam(initial_origin, lookat, up, fov, WIDTH, HEIGHT);
    // Wider framing keeps the bundled demonstration scenes in view throughout an orbit.
    if (animate) cam = Camera(Vec3(0, 0, 10), lookat, up, fov, WIDTH, HEIGHT);
    const Camera capture_camera = cam;
    const Light capture_light = light;
    int frame_index = 0;
    int exit_code = 0;

    bool running = true;
    auto last_time = std::chrono::high_resolution_clock::now();
    while (running) {
        auto current_time = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(current_time - last_time).count();  // delta time in seconds
        last_time = current_time;

        // Poll Events (Only handling Quit here)
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
        }

        // Continuous Input Handling
        const Uint8* state = SDL_GetKeyboardState(nullptr);

        // Calculate how much to rotate this specific frame
        float frame_rotation = animate ? 0.0f : ROTATION_SPEED * dt;

        // CAMERA CONTROLS
        if (state[SDL_SCANCODE_LEFT])  cam.rotateHorizontal(frame_rotation);
        if (state[SDL_SCANCODE_RIGHT]) cam.rotateHorizontal(-frame_rotation);
        if (state[SDL_SCANCODE_UP])    cam.rotateVertical(frame_rotation);
        if (state[SDL_SCANCODE_DOWN])  cam.rotateVertical(-frame_rotation);

        // LIGHT CONTROLS
        if (state[SDL_SCANCODE_A]) light.rotateHorizontal(frame_rotation);
        if (state[SDL_SCANCODE_D]) light.rotateHorizontal(-frame_rotation);
        if (state[SDL_SCANCODE_W]) light.rotateVertical(-frame_rotation);
        if (state[SDL_SCANCODE_S]) light.rotateVertical(frame_rotation);

        if (animate) {
            // Fixed angles produce a seamless loop independent of rendering speed.
            cam = capture_camera;
            light = capture_light;
            const float rotation = 2.0f * static_cast<float>(M_PI) * frame_index / frame_count;
            if (animate_camera) cam.rotateHorizontal(rotation);
            else light.rotateHorizontal(rotation);
        }

        if (use_gpu) {
            gpuRender(h_image, d_image, cam, light, d_tree, d_global_pool, d_global_stack, batch_size);
        }
        else {
            cpuRender(h_image, cam, light, h_tree);
        }
        updateSurface(surface, h_image);
        SDL_UpdateWindowSurface(window);
        if (animate) {
            const std::string number = std::to_string(frame_index);
            const auto path = std::filesystem::path(argv[5]) / ("frame_" + std::string(4 - number.size(), '0') + number + ".bmp");
            if (SDL_SaveBMP(surface, path.string().c_str()) != 0) {
                std::cerr << "[Error] Failed to save frame: " << SDL_GetError() << std::endl;
                exit_code = 1;
                running = false;
            }
            else {
                std::cout << "Saved frame " << ++frame_index << "/" << frame_count << std::endl;
                running = frame_index < frame_count;
            }
        }
        else if (output_file) {
            if (SDL_SaveBMP(surface, output_file) != 0) {
                std::cerr << "[Error] Failed to save render: " << SDL_GetError() << std::endl;
                exit_code = 1;
            }
            else {
                std::cout << "Saved render to '" << output_file << "'." << std::endl;
            }
            running = false;
        }
        else {
            std::cout << "Frame Time: " << (dt * 1000.0f) << " ms (" << (1.0f / dt) << " FPS)" << std::endl;
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();
    delete[] h_image;
    if (use_gpu) {
        checkCudaError(cudaFree(d_image), "cudaFree d_image");
        checkCudaError(cudaFree(d_global_pool), "cudaFree global pool");
        checkCudaError(cudaFree(d_global_stack), "cudaFree global stack");
        freeDeviceTree(d_tree);
    }
    freeHostTree(h_tree);
    return exit_code;
}
