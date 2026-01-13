#pragma once

#include <cstdint>
#include <cuda_runtime.h>


#include "rayCast.h"


constexpr size_t MAX_SHAPE_DATA_SIZE = 8;  // Largest shape size in number of floats

struct Span {
    float t_entry, t_exit;
    Hit entry_hit, exit_hit;
};

struct Sphere {
    Vec3 center;
    float radius;
    Material material;

    __host__ __device__ Sphere(const float* const data) : center(data[0], data[1], data[2]), radius(data[3]) {}

    __host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count) const {
        count = 0;
        Vec3 oc = ray.origin - center;
        float b = 2.0f * ray.dir.dot(oc);
        float c = oc.dot(oc) - radius * radius;
        float disc = b * b - 4.0f * c;
        if (disc < 0.0f) return;
        float sqrt_disc = sqrtf(disc);
        float t1 = (-b - sqrt_disc) * 0.5f;
        float t2 = (-b + sqrt_disc) * 0.5f;
        spans[0].t_entry = t1;
        spans[0].t_exit = t2;
        Vec3 p1 = ray.at(t1);
        spans[0].entry_hit.normal = (p1 - center).normalize();
        spans[0].entry_hit.mat = material;
        Vec3 p2 = ray.at(t2);
        spans[0].exit_hit.normal = (p2 - center).normalize();
        spans[0].exit_hit.mat = material;
        count = 1;
    }
};



