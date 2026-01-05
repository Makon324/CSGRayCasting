#pragma once

#include <vector>
#include <cmath>

#define INF 1e30f
#define EPS 1e-4f
#define MAX_DEPTH 5

struct Vec3 {
    float x, y, z;
    Vec3(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator*(float f) const { return Vec3(x * f, y * f, z * f); }
	Vec3 operator-() const { return Vec3(-x, -y, -z); }
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    float length() const { return std::sqrt(dot(*this)); }
    Vec3 normalize() const { float l = length(); return l > 0 ? Vec3(x / l, y / l, z / l) : *this; }
};

struct Color {
    float r, g, b;
    Color(float red = 0, float green = 0, float blue = 0) : r(red), g(green), b(blue) {}
    Color operator*(float f) const { return Color(r * f, g * f, b * f); }
    Color operator+(const Color& c) const { return Color(r + c.r, g + c.g, b + c.b); }
};

struct Ray {
    Vec3 origin, dir;
    Ray(const Vec3& o, const Vec3& d) : origin(o), dir(d.normalize()) {}
    Vec3 at(float t) const { return origin + dir * t; }
};

struct Material {
    Color color;           // Diffuse color
    float specular_coeff; // Specular coefficient (0-1)
    float shininess;      // Shininess exponent
};

struct Hit {
    Vec3 normal;          // Outward normal
    Material mat;        // Material at hit
public:
	Hit() = default;
	Hit(Vec3 n, Material m) : normal(n), mat(m) {}
};

struct Span {
    float t_entry, t_exit;
    Hit entry_hit, exit_hit;
};

class Shape {
public:
    virtual ~Shape() {}
    virtual std::vector<Span> getSpans(const Ray& ray) const = 0;
};

class Sphere : public Shape {
    Vec3 center;
    float radius;
    Material material;
public:
    Sphere(const Vec3& c, float r, const Material& m) : center(c), radius(r), material(m) {}
    std::vector<Span> getSpans(const Ray& ray) const override;
};

Vec3 reflect(const Vec3& i, const Vec3& n);  // Declaration