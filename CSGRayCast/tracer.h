#pragma once

#include <cstdint>
#include "rayCast.h"
#include "shape.h"
#include "csg.h"

constexpr uint32_t MAX_SPANS = 32;

// Forward declarations
struct Span;
struct Ray;
struct Light;
struct Color;
struct FlatCSGTree;

struct StackEntry {
    int start;
    uint32_t count;
};

// Device helper functions
__host__ __device__ void unionSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count);
__host__ __device__ void intersectionSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count);
__host__ __device__ void differenceSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count);

__host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count, const FlatCSGTree& tree, size_t node_idx, Span* thread_pool, StackEntry* thread_stack);
__host__ __device__ Color trace(const Ray& ray, const Light& light, const FlatCSGTree& tree, Span* thread_pool, StackEntry* thread_stack);

// UPDATED: Kernel now accepts offset and batch info for linear processing
__global__ void renderKernel(Color* image, const Camera cam, const Light light, const FlatCSGTree tree,
    Span* global_pool, StackEntry* global_stack,
    size_t pixel_offset, size_t batch_size, size_t total_pixels);

void copyTreeToDevice(const FlatCSGTree& h_tree, FlatCSGTree& d_tree);
void freeDeviceTree(FlatCSGTree& d_tree);
void freeHostTree(FlatCSGTree& tree);

void checkCudaError(cudaError_t err, const char* msg);








