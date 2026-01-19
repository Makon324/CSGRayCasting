#pragma once

#include <cmath>
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
    int material_id;

    __host__ __device__ Sphere(const float* const data) : center(data[0], data[1], data[2]), radius(data[3]), material_id(-1) {}

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
        spans[0].entry_hit.node_id = material_id;
        Vec3 p2 = ray.at(t2);
        spans[0].exit_hit.normal = (p2 - center).normalize();
        spans[0].exit_hit.node_id = material_id;
        count = 1;
    }
};

struct Cuboid {
    Vec3 min_pt;
    Vec3 max_pt;
    int material_id;

    // Data: x, y, z (corner), w, h, d
    __host__ __device__ Cuboid(const float* const data)
        : min_pt(data[0], data[1], data[2]),
        max_pt(data[0] + data[3], data[1] + data[4], data[2] + data[5]),
        material_id(-1) {
    }

    __host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count) const {
        count = 0;
        float t_min = -1e30f;
        float t_max = 1e30f;

        // Normals for entry and exit
        Vec3 n_entry(0, 0, 0);
        Vec3 n_exit(0, 0, 0);

        // Check X slab
        if (abs(ray.dir.x) < 1e-6f) {
            if (ray.origin.x < min_pt.x || ray.origin.x > max_pt.x) return;
        }
        else {
            float inv_d = 1.0f / ray.dir.x;
            float t1 = (min_pt.x - ray.origin.x) * inv_d;
            float t2 = (max_pt.x - ray.origin.x) * inv_d;
            Vec3 n_t1(-1, 0, 0);
            Vec3 n_t2(1, 0, 0);

            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; Vec3 tn = n_t1; n_t1 = n_t2; n_t2 = tn; }

            if (t1 > t_min) { t_min = t1; n_entry = n_t1; }
            if (t2 < t_max) { t_max = t2; n_exit = n_t2; }
        }

        // Check Y slab
        if (abs(ray.dir.y) < 1e-6f) {
            if (ray.origin.y < min_pt.y || ray.origin.y > max_pt.y) return;
        }
        else {
            float inv_d = 1.0f / ray.dir.y;
            float t1 = (min_pt.y - ray.origin.y) * inv_d;
            float t2 = (max_pt.y - ray.origin.y) * inv_d;
            Vec3 n_t1(0, -1, 0);
            Vec3 n_t2(0, 1, 0);

            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; Vec3 tn = n_t1; n_t1 = n_t2; n_t2 = tn; }

            if (t1 > t_min) { t_min = t1; n_entry = n_t1; }
            if (t2 < t_max) { t_max = t2; n_exit = n_t2; }
        }

        // Check Z slab
        if (abs(ray.dir.z) < 1e-6f) {
            if (ray.origin.z < min_pt.z || ray.origin.z > max_pt.z) return;
        }
        else {
            float inv_d = 1.0f / ray.dir.z;
            float t1 = (min_pt.z - ray.origin.z) * inv_d;
            float t2 = (max_pt.z - ray.origin.z) * inv_d;
            Vec3 n_t1(0, 0, -1);
            Vec3 n_t2(0, 0, 1);

            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; Vec3 tn = n_t1; n_t1 = n_t2; n_t2 = tn; }

            if (t1 > t_min) { t_min = t1; n_entry = n_t1; }
            if (t2 < t_max) { t_max = t2; n_exit = n_t2; }
        }

        if (t_min > t_max) return;

        spans[0].t_entry = t_min;
        spans[0].entry_hit.normal = n_entry;
        spans[0].entry_hit.node_id = material_id;

        spans[0].t_exit = t_max;
        spans[0].exit_hit.normal = n_exit;
        spans[0].exit_hit.node_id = material_id;
        count = 1;
    }
};

struct Cylinder {
    Vec3 center;
    float radius;
    float height;
    int material_id;

    // Data: x, y, z (bottom center), radius, height
    __host__ __device__ Cylinder(const float* const data)
        : center(data[0], data[1], data[2]), radius(data[3]), height(data[4]), material_id(-1) {
    }

    __host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count) const {
        count = 0;
        // Transform ray to local cylinder space (bottom at 0,0,0)
        Vec3 ro = ray.origin - center;
        Vec3 rd = ray.dir;

        float a = rd.x * rd.x + rd.z * rd.z;
        float b = 2.0f * (ro.x * rd.x + ro.z * rd.z);
        float c = ro.x * ro.x + ro.z * ro.z - radius * radius;

        float t_entries[2];
        Vec3 n_entries[2];
        int hit_count = 0;

        // Body intersection
        if (abs(a) > 1e-6f) {
            float disc = b * b - 4.0f * a * c;
            if (disc >= 0.0f) {
                float sqrt_disc = sqrtf(disc);
                float t1 = (-b - sqrt_disc) / (2.0f * a);
                float t2 = (-b + sqrt_disc) / (2.0f * a);

                float y1 = ro.y + t1 * rd.y;
                if (y1 >= 0.0f && y1 <= height) {
                    t_entries[hit_count] = t1;
                    n_entries[hit_count] = Vec3(ro.x + t1 * rd.x, 0.0f, ro.z + t1 * rd.z).normalize();
                    hit_count++;
                }
                float y2 = ro.y + t2 * rd.y;
                if (y2 >= 0.0f && y2 <= height) {
                    t_entries[hit_count] = t2;
                    n_entries[hit_count] = Vec3(ro.x + t2 * rd.x, 0.0f, ro.z + t2 * rd.z).normalize();
                    hit_count++;
                }
            }
        }

        // Cap intersections (y=0 and y=height)
        if (abs(rd.y) > 1e-6f) {
            // Bottom cap
            float t_bot = (0.0f - ro.y) / rd.y;
            Vec3 p_bot = ro + rd * t_bot;
            if (p_bot.x * p_bot.x + p_bot.z * p_bot.z <= radius * radius) {
                if (hit_count < 2) {
                    t_entries[hit_count] = t_bot;
                    n_entries[hit_count] = Vec3(0, -1, 0);
                    hit_count++;
                }
            }
            // Top cap
            float t_top = (height - ro.y) / rd.y;
            Vec3 p_top = ro + rd * t_top;
            if (p_top.x * p_top.x + p_top.z * p_top.z <= radius * radius) {
                if (hit_count < 2) {
                    t_entries[hit_count] = t_top;
                    n_entries[hit_count] = Vec3(0, 1, 0);
                    hit_count++;
                }
            }
        }

        if (hit_count < 2) return;

        // Sort hits (t_entry < t_exit)
        if (t_entries[0] > t_entries[1]) {
            float temp_t = t_entries[0]; t_entries[0] = t_entries[1]; t_entries[1] = temp_t;
            Vec3 temp_n = n_entries[0]; n_entries[0] = n_entries[1]; n_entries[1] = temp_n;
        }

        spans[0].t_entry = t_entries[0];
        spans[0].entry_hit.normal = n_entries[0];
        spans[0].entry_hit.node_id = material_id;

        spans[0].t_exit = t_entries[1];
        spans[0].exit_hit.normal = n_entries[1];
        spans[0].exit_hit.node_id = material_id;
        count = 1;
    }
};

struct Cone {
    Vec3 center;
    float radius;
    float height;
    int material_id;

    // Data: x, y, z (bottom center), radius, height
    __host__ __device__ Cone(const float* const data)
        : center(data[0], data[1], data[2]), radius(data[3]), height(data[4]), material_id(-1) {
    }

    __host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count) const {
        count = 0;
        Vec3 ro = ray.origin - center;
        Vec3 rd = ray.dir;

        float k = radius / height;
        float k2 = k * k;

        // x^2 + z^2 = k^2 * (h - y)^2
        float a = rd.x * rd.x + rd.z * rd.z - k2 * rd.y * rd.y;
        float b = 2.0f * (ro.x * rd.x + ro.z * rd.z + k2 * (height - ro.y) * rd.y);
        float c = ro.x * ro.x + ro.z * ro.z - k2 * (height - ro.y) * (height - ro.y);

        float t_entries[2];
        Vec3 n_entries[2];
        int hit_count = 0;

        // Body intersection
        if (abs(a) > 1e-6f) {
            float disc = b * b - 4.0f * a * c;
            if (disc >= 0.0f) {
                float sqrt_disc = sqrtf(disc);
                float t1 = (-b - sqrt_disc) / (2.0f * a);
                float t2 = (-b + sqrt_disc) / (2.0f * a);

                auto check_body = [&](float t) {
                    float y = ro.y + t * rd.y;
                    if (y >= 0.0f && y <= height) {
                        t_entries[hit_count] = t;
                        // Gradient normal
                        Vec3 p = ro + rd * t;
                        float n_y = k2 * (height - p.y); // Derived from derivative
                        n_entries[hit_count] = Vec3(p.x, n_y, p.z).normalize();
                        hit_count++;
                    }
                    };

                check_body(t1);
                if (hit_count < 2) check_body(t2);
            }
        }

        // Base Cap intersection (y=0)
        if (abs(rd.y) > 1e-6f) {
            float t_base = (0.0f - ro.y) / rd.y;
            Vec3 p_base = ro + rd * t_base;
            if (p_base.x * p_base.x + p_base.z * p_base.z <= radius * radius) {
                if (hit_count < 2) {
                    t_entries[hit_count] = t_base;
                    n_entries[hit_count] = Vec3(0, -1, 0);
                    hit_count++;
                }
            }
        }

        if (hit_count < 2) return;

        // Sort hits
        if (t_entries[0] > t_entries[1]) {
            float temp_t = t_entries[0]; t_entries[0] = t_entries[1]; t_entries[1] = temp_t;
            Vec3 temp_n = n_entries[0]; n_entries[0] = n_entries[1]; n_entries[1] = temp_n;
        }

        spans[0].t_entry = t_entries[0];
        spans[0].entry_hit.normal = n_entries[0];
        spans[0].entry_hit.node_id = material_id;

        spans[0].t_exit = t_entries[1];
        spans[0].exit_hit.normal = n_entries[1];
        spans[0].exit_hit.node_id = material_id;
        count = 1;
    }
};