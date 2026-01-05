#include "scene.h"
#include <fstream>

// Calculates color using Phong model
Color Scene::phongShade(const Vec3& point, const Vec3& normal, const Material mat, const Vec3& view_dir) const {
    Vec3 l_dir = light.direction;

    // Shadow ray
    Ray shadow_ray(point + normal * EPS, l_dir);
    auto shadow_spans = getRoot()->getSpans(shadow_ray);
    bool in_shadow = false;
    for (const auto& s : shadow_spans) {
        if (s.t_entry > EPS) {
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

void Scene::renderCPU(const std::string& filename) {
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

void Scene::renderGPU(const std::string& filename) {
    HostFlatCSGTree host_flat = flattenTree();
    size_t num_nodes = host_flat.nodes.size();
    if (num_nodes == 0) return;

    FlatCSGTree d_tree;

    cudaMalloc(&d_tree.nodes, num_nodes * sizeof(FlatCSGNodeInfo));
    cudaMemcpy(d_tree.nodes, host_flat.nodes.data(), num_nodes * sizeof(FlatCSGNodeInfo), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tree.data, num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float));
    cudaMemcpy(d_tree.data, host_flat.data.data(), num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tree.red, num_nodes * sizeof(float));
    cudaMemcpy(d_tree.red, host_flat.red.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tree.green, num_nodes * sizeof(float));
    cudaMemcpy(d_tree.green, host_flat.green.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tree.blue, num_nodes * sizeof(float));
    cudaMemcpy(d_tree.blue, host_flat.blue.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tree.specular_coeff, num_nodes * sizeof(float));
    cudaMemcpy(d_tree.specular_coeff, host_flat.specular_coeff.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tree.shininess, num_nodes * sizeof(float));
    cudaMemcpy(d_tree.shininess, host_flat.shininess.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tree.left_indexes, num_nodes * sizeof(size_t));
    cudaMemcpy(d_tree.left_indexes, host_flat.left_indexes.data(), num_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_tree.right_indexes, num_nodes * sizeof(size_t));
    cudaMemcpy(d_tree.right_indexes, host_flat.right_indexes.data(), num_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

    d_tree.num_nodes = num_nodes;

    int width = camera.getWidth();
    int height = camera.getHeight();

    // Compute shared mem size
    size_t shared_size = num_nodes * (sizeof(FlatCSGNodeInfo) + MAX_SHAPE_DATA_SIZE * sizeof(float) + 5 * sizeof(float) + 2 * sizeof(size_t));

    unsigned char* d_output;
    cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    renderKernel << <grid, block, shared_size >> > (d_output, width, height, camera, light, d_tree);

    unsigned char* h_output = new unsigned char[width * height * 3];
    cudaMemcpy(h_output, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Write PPM
    std::ofstream out(filename);
    if (out) {
        out << "P3\n" << width << " " << height << "\n255\n";
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                size_t idx = (j * width + i) * 3;
                out << (int)h_output[idx + 0] << " " << (int)h_output[idx + 1] << " " << (int)h_output[idx + 2] << "\n";
            }
        }
        out.close();
    }

    delete[] h_output;
    cudaFree(d_output);
    cudaFree(d_tree.nodes);
    cudaFree(d_tree.data);
    cudaFree(d_tree.red);
    cudaFree(d_tree.green);
    cudaFree(d_tree.blue);
    cudaFree(d_tree.specular_coeff);
    cudaFree(d_tree.shininess);
    cudaFree(d_tree.left_indexes);
    cudaFree(d_tree.right_indexes);
}

size_t flattenHelper(const CSGNode* node, HostFlatCSGTree& flat) {
    if (!node) return size_t(-1);

    size_t left_idx = flattenHelper(node->left, flat);
    size_t right_idx = flattenHelper(node->right, flat);

    size_t curr_idx = flat.nodes.size();

    FlatCSGNodeInfo info;
    info.op = node->op;

    if (node->shape) {
        info.shape_type = ShapeType::Sphere;
        Sphere* s = dynamic_cast<Sphere*>(node->shape);
        flat.data.push_back(s->center.x);
        flat.data.push_back(s->center.y);
        flat.data.push_back(s->center.z);
        flat.data.push_back(s->radius);
        for (size_t pad = 4; pad < MAX_SHAPE_DATA_SIZE; ++pad) flat.data.push_back(0.0f);
        flat.red.push_back(s->material.color.r);
        flat.green.push_back(s->material.color.g);
        flat.blue.push_back(s->material.color.b);
        flat.specular_coeff.push_back(s->material.specular_coeff);
        flat.shininess.push_back(s->material.shininess);
        flat.left_indexes.push_back(size_t(-1));
        flat.right_indexes.push_back(size_t(-1));
    }
    else {
        info.shape_type = ShapeType::TreeNode;
        for (size_t pad = 0; pad < MAX_SHAPE_DATA_SIZE; ++pad) flat.data.push_back(0.0f);
        flat.red.push_back(0.0f);
        flat.green.push_back(0.0f);
        flat.blue.push_back(0.0f);
        flat.specular_coeff.push_back(0.0f);
        flat.shininess.push_back(0.0f);
        flat.left_indexes.push_back(left_idx);
        flat.right_indexes.push_back(right_idx);
    }

    flat.nodes.push_back(info);
    return curr_idx;
}

HostFlatCSGTree Scene::flattenTree() const {
    HostFlatCSGTree flat;
    flattenHelper(root, flat);
    return flat;
}