#include <iostream>
#include <cstring>
#include <cmath>
#include <SDL.h>
#include <chrono>
#include <cuda_runtime.h>

#include "csg.h"
#include "rayCast.h"
#include "tracer.h"

// --- CONSTANTS ---
constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;
constexpr float LIGHT_ROTATION_SPEED = 0.02f;
constexpr size_t MAX_SCRATCH_MEMORY_BYTES = 512ULL * 1024ULL * 1024ULL;

constexpr int threadsPerBlock = 256;

void cpuRender(Color* h_image, const Camera& cam, const Light& light, const FlatCSGTree& tree) {
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float s = (x + 0.5f) / WIDTH;
            float t = (y + 0.5f) / HEIGHT;
            Ray ray = cam.getRay(s, t);
            h_image[y * WIDTH + x] = trace(ray, light, tree, nullptr, nullptr);
        }
    }
}

void gpuRender(Color* h_image, Color* d_image, const Camera& cam, const Light& light, const FlatCSGTree& d_tree,
    Span* d_global_pool, StackEntry* d_global_stack, size_t batch_size) {

    // Shared memory size calculation
    size_t shared_size = d_tree.num_nodes * (sizeof(FlatCSGNodeInfo) + MAX_SHAPE_DATA_SIZE * sizeof(float) + 6 * sizeof(float) + 3 * sizeof(size_t));

    size_t total_pixels = static_cast<size_t>(WIDTH) * HEIGHT;    

    // BATCH LOOP
    // Instead of one giant launch, we iterate through pixels in chunks of 'batch_size'
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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [cpu|gpu] [scene_file]\n";
        return 1;
    }
    bool use_gpu = (std::strcmp(argv[1], "gpu") == 0);

    FlatCSGTree h_tree;
    try {
        h_tree = loadFromFile(argv[2]);
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] Failed to load scene file: " << e.what() << std::endl;
        return 1;
    }

    // CRITICAL: Check for empty tree to prevent Division By Zero later
    if (h_tree.num_nodes == 0) {
        std::cerr << "[Error] The scene file is empty or contains no valid nodes." << std::endl;
        return 1;
    }

    // Compute sizes based on tree
    h_tree.max_pool_size = h_tree.num_nodes * MAX_SPANS * 2;
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

        // --- SMART MEMORY ALLOCATION ---
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
    SDL_Window* window = SDL_CreateWindow("CSG Ray Tracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);
    SDL_Surface* surface = SDL_GetWindowSurface(window);
    float angle = 0.0f;
    Vec3 initial_origin(5.0f * sinf(angle), 0.0f, 5.0f * cosf(angle));
    Camera cam(initial_origin, lookat, up, fov, WIDTH, HEIGHT);

    bool running = true;
    int frame = 0;
    while (running) {
        auto start = std::chrono::high_resolution_clock::now();
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_LEFT) cam.rotateY(0.1f);
                if (event.key.keysym.sym == SDLK_RIGHT) cam.rotateY(-0.1f);
                if (event.key.keysym.sym == SDLK_UP) cam.rotateVertical(0.1f);
                if (event.key.keysym.sym == SDLK_DOWN) cam.rotateVertical(-0.1f);
            }
        }

        light.rotateY(LIGHT_ROTATION_SPEED);

        if (use_gpu) {
            gpuRender(h_image, d_image, cam, light, d_tree, d_global_pool, d_global_stack, batch_size);
        }
        else {
            cpuRender(h_image, cam, light, h_tree);
        }
        updateSurface(surface, h_image);
        SDL_UpdateWindowSurface(window);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        ++frame;
        std::cout << "Frame " << frame << ": " << duration << " ms" << std::endl;
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
    return 0;
}