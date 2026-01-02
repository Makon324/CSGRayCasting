#include "scene.h"
#include <fstream>

// Calculates color using Phong model
Color Scene::phongShade(const Vec3& point, const Vec3& normal, const Material mat, const Vec3& view_dir) const {
    Vec3 l_dir = (light.position - point).normalize();
    float light_dist = (light.position - point).length();

    // Shadow ray
    Ray shadow_ray(point + normal * EPS, l_dir);
    auto shadow_spans = getRoot()->getSpans(shadow_ray);
    bool in_shadow = false;
    for (const auto& s : shadow_spans) {
        if (s.t_entry > EPS && s.t_entry < light_dist) {
            in_shadow = true;
            break;
        }
    }

    // Ambient
    Color ambient = mat.color * 0.1f;

    if (in_shadow) return ambient;

    // Diffuse
    float diff = std::max(0.0f, normal.dot(l_dir));
    Color diffuse = mat.color * diff;

    // Specular (classic Phong reflection vector)
    Vec3 r = reflect(-l_dir, normal);  // Reflection of light direction over normal
    float spec = std::pow(std::max(0.0f, view_dir.dot(r)), mat.shininess) * mat.specular_coeff;
    Color specular = Color(1.0f, 1.0f, 1.0f) * spec;  // White highlights

    return ambient + diffuse + specular;
}

// trace ray and return color
Color Scene::trace(const Ray& ray) const {
    auto spans = getRoot()->getSpans(ray);
    if (spans.empty()) return Color(0, 0, 0);  // Background

    // Find closest entry hit
    float min_t = INF;
    Hit closest_hit;
    for (const auto& span : spans) {
        if (span.t_entry > EPS && span.t_entry < min_t) {
            min_t = span.t_entry;
            closest_hit = span.entry_hit;
        }
    }
    if (min_t == INF) return Color(0, 0, 0);

    Vec3 point = ray.at(min_t);
    Vec3 normal = closest_hit.normal;
    Vec3 view_dir = -ray.dir;  // From point to viewer? No, Phong expects to viewer, so -ray.dir (since ray.dir is away from viewer)

    Color color = phongShade(point, normal, closest_hit.mat, view_dir);

    return color;
}

void Scene::render(const std::string& filename) {
    std::ofstream out(filename);
    if (!out) return;
    int width = camera.getWidth();
    int height = camera.getHeight();
    out << "P3\n" << width << " " << height << "\n255\n";
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float s = (i + 0.5f) / width;
            float t = (j + 0.5f) / height;
            Ray r = camera.getRay(s, t);
            Color color = trace(r);
            // Clamp and gamma-correct if desired; simple clamp here
            int ir = std::min(255, static_cast<int>(255.99f * std::max(0.0f, color.r)));
            int ig = std::min(255, static_cast<int>(255.99f * std::max(0.0f, color.g)));
            int ib = std::min(255, static_cast<int>(255.99f * std::max(0.0f, color.b)));
            out << ir << " " << ig << " " << ib << "\n";
        }
    }
    out.close();
}

