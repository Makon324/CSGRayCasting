#include "shape.h"
#include <cmath>
#include <algorithm>

std::vector<Span> Sphere::getSpans(const Ray& ray) const {
    std::vector<Span> spans;
    Vec3 oc = ray.origin - center;
    float a = ray.dir.dot(ray.dir);
    float b = 2 * oc.dot(ray.dir);
    float c = oc.dot(oc) - radius * radius;

    float disc = b * b - 4 * a * c;
    if (disc < -1e-6f) return spans;  // Reject only clearly negative
    if (disc < 0.0f) disc = 0.0f;     // Clamp small negatives to zero (treat as tangent)

    float sqrt_disc = std::sqrt(disc);

    float t1 = (-b - sqrt_disc) / (2 * a);
    float t2 = (-b + sqrt_disc) / (2 * a);
    if (t1 > t2) std::swap(t1, t2);
    if (t2 < 0) return spans;
    if (t1 < 0) t1 = 0;  // Clip if ray starts inside (simplified handling)


    Span s;
    s.t_entry = t1;
    Vec3 entry_point = ray.at(t1);
    s.entry_hit.normal = (entry_point - center).normalize();
    s.entry_hit.mat = material;
    s.t_exit = t2;
    Vec3 exit_point = ray.at(t2);
    s.exit_hit.normal = (exit_point - center).normalize();
    s.exit_hit.mat = material;
    spans.push_back(s);
    return spans;
}


