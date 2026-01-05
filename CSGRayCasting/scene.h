#pragma once

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "shape.h"
#include "csg.h"
#include <vector>
#include <string>
#include <cmath>



#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif



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
        float theta = (float) (fov * M_PI / 180.0f);
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

    __host__ __device__ int getWidth() const { return width; }
    __host__ __device__ int getHeight() const { return height; }
};


struct Light {
    Vec3 direction;
    __host__ __device__ Light(const Vec3& dir) : direction(dir.normalize()) {}
};

struct HostFlatCSGTree {
    std::vector<FlatCSGNodeInfo> nodes;
    std::vector<float> data;
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
    std::vector<float> specular_coeff;
    std::vector<float> shininess;
    std::vector<size_t> left_indexes;
    std::vector<size_t> right_indexes;
};

class Scene {
    CSGNode* root;
    Light light;
    Camera camera;
public:
    Scene(CSGNode* r, const Light& l, const Camera& c)
        : root(r), light(l), camera(c) {
    }
    void renderCPU(const std::string& filename);  // Outputs PPM image
	void renderGPU(const std::string& filename);  // Outputs PPM image
    HostFlatCSGTree flattenTree() const;
    CSGNode* getRoot() const { return root; }
    Color trace(const Ray& ray) const;
    Color phongShade(const Vec3& point, const Vec3& normal, const Material mat, const Vec3& view_dir) const;
};

__global__ void renderKernel(unsigned char* d_output, int width, int height, const Camera cam,
    const Light light, const FlatCSGTree d_tree);

