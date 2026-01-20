#pragma once

#include <cstdint>
#include "rayCast.h"
#include "shape.h"
#include "csg.h"


struct StackEntry {
    int start;
    uint32_t count;
};

// --- STRIDED MEMORY WRAPPERS ---
// Encapsulates the logic: Address = Base + (Index * Stride)
struct StridedSpan {
    Span* data;
    size_t stride;

    __host__ __device__ StridedSpan(Span* d = nullptr, size_t s = 1) : data(d), stride(s) {}

    __host__ __device__ Span& operator[](int i) const {
        return data[i * stride];
    }

    // Get raw pointer to specific element (needed for Sphere::getSpans)
    __host__ __device__ Span* at(int i) const {
        return &data[i * stride];
    }

    __host__ __device__ bool is_valid() const { return data != nullptr; }
};

struct StridedStack {
    StackEntry* data;
    size_t stride;

    __host__ __device__ StridedStack(StackEntry* d = nullptr, size_t s = 1) : data(d), stride(s) {}

    __host__ __device__ StackEntry& operator[](int i) const {
        return data[i * stride];
    }

    __host__ __device__ bool is_valid() const { return data != nullptr; }
};


__host__ __device__ void unionSpans(const StridedSpan& left, uint32_t left_count, const StridedSpan& right, uint32_t right_count, StridedSpan& result, uint32_t& result_count);
__host__ __device__ void intersectionSpans(const StridedSpan& left, uint32_t left_count, const StridedSpan& right, uint32_t right_count, StridedSpan& result, uint32_t& result_count);
__host__ __device__ void differenceSpans(const StridedSpan& left, uint32_t left_count, const StridedSpan& right, uint32_t right_count, StridedSpan& result, uint32_t& result_count);

// CHANGED: getSpans now returns the start index via pointer, no local buffer copy needed
__host__ __device__ void getSpans(const Ray& ray, size_t* out_start_idx, uint32_t* out_count, const FlatCSGTree& tree, size_t node_idx, StridedSpan thread_pool, StridedStack thread_stack);
__host__ __device__ Color trace(const Ray& ray, const Light& light, const FlatCSGTree& tree, StridedSpan thread_pool, StridedStack thread_stack);

__global__ void renderKernel(Color* image, const Camera cam, const Light light, const FlatCSGTree tree,
    Span* global_pool, StackEntry* global_stack,
    size_t pixel_offset, size_t batch_size, size_t total_pixels);

void copyTreeToDevice(const FlatCSGTree& h_tree, FlatCSGTree& d_tree);
void freeDeviceTree(FlatCSGTree& d_tree);
void freeHostTree(FlatCSGTree& tree);

void checkCudaError(cudaError_t err, const char* msg);



