#pragma once

#include "rayCast.h"
#include "shape.h"
#include "csg.h"

constexpr uint32_t MAX_SPANS = 32;  // Reduced for memory efficiency
constexpr int MAX_DEPTH = 32;       // Reduced for memory efficiency

struct Span;
struct Ray;
struct Light;
struct Color;
struct FlatCSGTree;

__host__ __device__ void unionSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count);
__host__ __device__ void intersectionSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count);
__host__ __device__ void differenceSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count);
__host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count, const FlatCSGTree& tree, size_t node_idx);
__host__ __device__ Color trace(const Ray& ray, const Light& light, const FlatCSGTree& tree);

__global__ void renderKernel(Color* image, const Camera cam, const Light light, const FlatCSGTree tree);

void copyTreeToDevice(const FlatCSGTree& h_tree, FlatCSGTree& d_tree);
void freeDeviceTree(FlatCSGTree& d_tree);
void freeHostTree(FlatCSGTree& tree);





