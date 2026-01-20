#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator*(float f) const { return Vec3(x * f, y * f, z * f); }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 cross(const Vec3& v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    __host__ __device__ float length() const { return std::sqrt(dot(*this)); }
    __host__ __device__ Vec3 normalize() const { float l = length(); return l > 0 ? Vec3(x / l, y / l, z / l) : *this; }
};

struct Ray {
    Vec3 origin;
	Vec3 dir;
    __host__ __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), dir(d.normalize()) {}
    __host__ __device__ Vec3 at(float t) const { return origin + dir * t; }
};

struct Color {
    float r, g, b;
    __host__ __device__ Color(float red = 0, float green = 0, float blue = 0) : r(red), g(green), b(blue) {}
    __host__ __device__ Color operator*(float f) const { return Color(r * f, g * f, b * f); }
    __host__ __device__ Color operator+(const Color& c) const { return Color(r + c.r, g + c.g, b + c.b); }
};

struct Material {
    Color color;
    float diffuse_coeff;
    float specular_coeff;
    float shininess;
    __host__ __device__ Material() : color(0), diffuse_coeff(0), specular_coeff(0), shininess(0) {}
    __host__ __device__ Material(Color c, float dc, float sc, float sh) : color(c), diffuse_coeff(dc), specular_coeff(sc), shininess(sh) {}
};

struct Hit {
    Vec3 normal;           // Outward normal
    uint32_t node_id;      // Index into the FlatCSGTree material arrays

    __host__ __device__ Hit() : normal(0), node_id(0) {}
    __host__ __device__ Hit(Vec3 n, uint32_t id) : normal(n), node_id(id) {}
};

struct Camera {
    Vec3 origin, lookat, up;
    float fov;  // Vertical field of view in degrees
    int width, height;
    Vec3 u, v, w;  // Orthonormal basis
    float aspect;
    float viewport_height, viewport_width;

    __host__ __device__ Camera(const Vec3& o, const Vec3& la, const Vec3& u_vec, float _fov, int _width, int _height)
        : origin(o), lookat(la), up(u_vec), fov(_fov), width(_width), height(_height) {
        aspect = static_cast<float>(_width) / _height;
        float theta = (float)(fov * M_PI / 180.0f);
        viewport_height = 2.0f * tan(theta / 2.0f);
        viewport_width = aspect * viewport_height;

        w = (origin - lookat).normalize();
        u = up.cross(w).normalize();
        v = w.cross(u);
    }

    __host__ __device__ Ray getRay(float s, float t) const {  // s,t in [0,1] for pixel centers
        Vec3 rd = Vec3((s - 0.5f) * viewport_width, (0.5f - t) * viewport_height, -1.0f);
        Vec3 dir = u * rd.x + v * rd.y + w * rd.z;
        return Ray(origin, dir.normalize());
    }

    __host__ __device__ void updateBasis() {
        w = (origin - lookat).normalize();
        u = up.cross(w).normalize();
        v = w.cross(u);
    }

    __host__ __device__ void rotateHorizontal(float delta_angle) {
        Vec3 dir = origin - lookat;
        float r_xz = sqrtf(dir.x * dir.x + dir.z * dir.z);

        float current_angle = atan2(dir.z, dir.x);
        float new_angle = current_angle + delta_angle;

        origin.x = lookat.x + r_xz * cosf(new_angle);
        origin.z = lookat.z + r_xz * sinf(new_angle);

        updateBasis();
    }

    __host__ __device__ void rotateVertical(float delta_angle) {
        Vec3 dir = origin - lookat;
        float r = dir.length();

        float current_pitch = asinf(dir.y / r);
        float new_pitch = current_pitch + delta_angle;

        // Clamp pitch
        const float LIMIT = 89.0f * (static_cast<float>(M_PI) / 180.0f);
        if (new_pitch > LIMIT) new_pitch = LIMIT;
        if (new_pitch < -LIMIT) new_pitch = -LIMIT;

        origin.y = lookat.y + r * sinf(new_pitch);
        float r_xz = r * cosf(new_pitch);

        float theta = atan2(dir.z, dir.x);
        origin.x = lookat.x + r_xz * cosf(theta);
        origin.z = lookat.z + r_xz * sinf(theta);

        updateBasis();
    }

    __host__ __device__ int getWidth() const { return width; }
    __host__ __device__ int getHeight() const { return height; }
};

struct Light {
    Vec3 direction;
    __host__ __device__ Light(const Vec3& dir) : direction(dir.normalize()) {}

    __host__ __device__ void rotateHorizontal(float delta_angle) {
        float r_xz = sqrtf(direction.x * direction.x + direction.z * direction.z);

        float current_angle = atan2(direction.z, direction.x);
        float new_angle = current_angle + delta_angle;

        direction.x = r_xz * cosf(new_angle);
        direction.z = r_xz * sinf(new_angle);

        direction = direction.normalize();
    }

    __host__ __device__ void rotateVertical(float delta_angle) {
        float r = direction.length();
        float current_pitch = asinf(direction.y / r);
        float new_pitch = current_pitch + delta_angle;

        const float LIMIT = 89.0f * (static_cast<float>(M_PI) / 180.0f);
        if (new_pitch > LIMIT) new_pitch = LIMIT;
        if (new_pitch < -LIMIT) new_pitch = -LIMIT;

        direction.y = r * sinf(new_pitch);
        float r_xz = r * cosf(new_pitch);

        float theta = atan2f(direction.z, direction.x);
        direction.x = r_xz * cosf(theta);
        direction.z = r_xz * sinf(theta);

        direction = direction.normalize();
    }
};