#include "csg.h"
#include "rayCast.h"
#include "shape.h"
#include "tracer.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BenchmarkResult {
    std::string scene;
    size_t nodes;
    size_t primitives;
    double load_ms;
    double frame_ms;
    double mrays_per_second;
    double checksum;
};

void compactTreeMemory(FlatCSGTree& tree) {
    size_t primitive_count = 0;
    for (size_t i = 0; i < tree.num_nodes; ++i) {
        if (tree.nodes[i].shape_type != ShapeType::TreeNode) {
            tree.primitive_idx[i] = static_cast<int32_t>(primitive_count++);
        }
        else {
            tree.primitive_idx[i] = -1;
        }
    }
    tree.num_primitives = primitive_count;

    float* compact_data = new float[primitive_count * MAX_SHAPE_DATA_SIZE];
    float* compact_red = new float[primitive_count];
    float* compact_green = new float[primitive_count];
    float* compact_blue = new float[primitive_count];
    float* compact_diffuse = new float[primitive_count];
    float* compact_specular = new float[primitive_count];
    float* compact_shininess = new float[primitive_count];

    for (size_t node_index = 0; node_index < tree.num_nodes; ++node_index) {
        int32_t primitive_index = tree.primitive_idx[node_index];
        if (primitive_index < 0) {
            continue;
        }

        std::memcpy(
            &compact_data[primitive_index * MAX_SHAPE_DATA_SIZE],
            &tree.data[node_index * MAX_SHAPE_DATA_SIZE],
            MAX_SHAPE_DATA_SIZE * sizeof(float));
        compact_red[primitive_index] = tree.red[node_index];
        compact_green[primitive_index] = tree.green[node_index];
        compact_blue[primitive_index] = tree.blue[node_index];
        compact_diffuse[primitive_index] = tree.diffuse_coeff[node_index];
        compact_specular[primitive_index] = tree.specular_coeff[node_index];
        compact_shininess[primitive_index] = tree.shininess[node_index];
    }

    delete[] tree.data;
    delete[] tree.red;
    delete[] tree.green;
    delete[] tree.blue;
    delete[] tree.diffuse_coeff;
    delete[] tree.specular_coeff;
    delete[] tree.shininess;

    tree.data = compact_data;
    tree.red = compact_red;
    tree.green = compact_green;
    tree.blue = compact_blue;
    tree.diffuse_coeff = compact_diffuse;
    tree.specular_coeff = compact_specular;
    tree.shininess = compact_shininess;
}

int parsePositive(const char* value, const char* name) {
    int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

BenchmarkResult benchmarkScene(
    const std::filesystem::path& repository_root,
    const std::string& scene,
    int iterations,
    int width,
    int height) {
    auto load_start = std::chrono::steady_clock::now();

    std::string scene_path = (repository_root / scene).string();
    FlatCSGTree tree = loadFromFile(scene_path.c_str());
    compactTreeMemory(tree);
    tree.max_pool_size = computeTotalSpanUsage(tree);
    tree.max_stack_depth = computeMaxDepth(tree) * 2;

    auto load_end = std::chrono::steady_clock::now();

    if (tree.max_pool_size == 0 || tree.max_stack_depth == 0) {
        freeHostTree(tree);
        throw std::runtime_error("invalid scratch-buffer size for " + scene);
    }

    std::vector<Span> pool_data(tree.max_pool_size);
    std::vector<StackEntry> stack_data(tree.max_stack_depth);
    StridedSpan pool(pool_data.data());
    StridedStack stack(stack_data.data());

    Camera camera(
        Vec3(0, 0, 5),
        Vec3(0, 0, 0),
        Vec3(0, 1, 0),
        60.0f,
        width,
        height);
    Light light(Vec3(1, 1, 1));

    auto renderFrame = [&]() {
        double checksum = 0.0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float s = (x + 0.5f) / static_cast<float>(width);
                float t = (y + 0.5f) / static_cast<float>(height);
                Color color = trace(
                    camera.getRay(s, t),
                    light,
                    tree,
                    pool,
                    stack);
                checksum += color.r + color.g * 3.0 + color.b * 7.0;
            }
        }
        return checksum;
    };

    volatile double warmup_checksum = renderFrame();
    (void)warmup_checksum;

    double checksum = 0.0;
    auto render_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        checksum += renderFrame();
    }
    auto render_end = std::chrono::steady_clock::now();

    double load_ms =
        std::chrono::duration<double, std::milli>(load_end - load_start).count();
    double total_render_ms =
        std::chrono::duration<double, std::milli>(render_end - render_start).count();
    double frame_ms = total_render_ms / iterations;
    double ray_count =
        static_cast<double>(width) * height * iterations;
    double mrays_per_second = ray_count / (total_render_ms / 1000.0) / 1.0e6;

    BenchmarkResult result{
        scene,
        tree.num_nodes,
        tree.num_primitives,
        load_ms,
        frame_ms,
        mrays_per_second,
        checksum
    };

    freeHostTree(tree);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr
            << "Usage: CSGRayCastBenchmarks <repository_root> "
            << "[iterations] [width] [height]\n";
        return 2;
    }

    try {
        std::filesystem::path repository_root(argv[1]);
        int iterations = argc >= 3 ? parsePositive(argv[2], "iterations") : 3;
        int width = argc >= 4 ? parsePositive(argv[3], "width") : 160;
        int height = argc >= 5 ? parsePositive(argv[4], "height") : 120;

        const std::vector<std::string> scenes{
            "single_sphere.txt",
            "complex_scene.txt",
            "large_city.txt"
        };

        std::cout << "# iterations=" << iterations
                  << ",width=" << width
                  << ",height=" << height << "\n";
        std::cout
            << "scene,nodes,primitives,load_ms,frame_ms,"
            << "mrays_per_second,checksum\n";
        std::cout << std::fixed << std::setprecision(3);

        for (const std::string& scene : scenes) {
            BenchmarkResult result = benchmarkScene(
                repository_root, scene, iterations, width, height);
            std::cout
                << result.scene << ","
                << result.nodes << ","
                << result.primitives << ","
                << result.load_ms << ","
                << result.frame_ms << ","
                << result.mrays_per_second << ","
                << result.checksum << "\n";
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[Error] " << error.what() << "\n";
        return 1;
    }

    return 0;
}
