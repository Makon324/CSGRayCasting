#include <iostream>
#include <cstring>
#include <cmath>
#include <SDL.h>
#include <chrono>

#include "csg.h"
#include "rayCast.h"
#include "tracer.h"


void cpuRender(Color* h_image, const Camera& cam, const Light& light, const FlatCSGTree& tree, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float s = (x + 0.5f) / width;
            float t = (y + 0.5f) / height;
            Ray ray = cam.getRay(s, t);
            h_image[y * width + x] = trace(ray, light, tree);
        }
    }
}

void gpuRender(Color* h_image, Color* d_image, const Camera& cam, const Light& light, const FlatCSGTree& d_tree, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    size_t shared_size = d_tree.num_nodes * (sizeof(FlatCSGNodeInfo) + MAX_SHAPE_DATA_SIZE * sizeof(float) + 6 * sizeof(float) + 2 * sizeof(size_t));
    renderKernel << <grid, block, shared_size >> > (d_image, cam, light, d_tree);
    cudaDeviceSynchronize();
    cudaMemcpy(h_image, d_image, width * height * sizeof(Color), cudaMemcpyDeviceToHost);
}

void updateSurface(SDL_Surface* surface, Color* h_image, int width, int height) {
    Uint8* pixels = static_cast<Uint8*>(surface->pixels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Color c = h_image[y * width + x];
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
    FlatCSGTree h_tree = loadFromFile(argv[2]);
    int width = 800;
    int height = 600;
    Vec3 lookat(0, 0, 0);
    Vec3 up(0, 1, 0);
    float fov = 60.0f;
    Light light(Vec3(1, 1, 1));
    Color* h_image = new Color[width * height];
    Color* d_image = nullptr;
    FlatCSGTree d_tree;
    if (use_gpu) {
        cudaMalloc(&d_image, width * height * sizeof(Color));
        copyTreeToDevice(h_tree, d_tree);
    }
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("CSG Ray Tracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, 0);
    SDL_Surface* surface = SDL_GetWindowSurface(window);
    float angle = 0.0f;
    Vec3 initial_origin(5.0f * sinf(angle), 0.0f, 5.0f * cosf(angle));
    Camera cam(initial_origin, lookat, up, fov, width, height);
    bool running = true;
    int frame = 0;
    while (running) {
        auto start = std::chrono::high_resolution_clock::now();
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_LEFT) {
                    cam.rotateY(0.1f);
                }
                if (event.key.keysym.sym == SDLK_RIGHT) {
                    cam.rotateY(-0.1f);
                }
            }
        }
        if (use_gpu) {
            gpuRender(h_image, d_image, cam, light, d_tree, width, height);
        }
        else {
            cpuRender(h_image, cam, light, h_tree, width, height);
        }
        updateSurface(surface, h_image, width, height);
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
        cudaFree(d_image);
        freeDeviceTree(d_tree);
    }
    freeHostTree(h_tree);
    return 0;
}

