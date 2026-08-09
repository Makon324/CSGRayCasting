#include "csg.h"
#include "rayCast.h"
#include "shape.h"
#include "tracer.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failure_count = 0;
int test_count = 0;
std::string current_test;
std::filesystem::path repository_root;

bool nearlyEqual(float actual, float expected, float tolerance = 1e-4f) {
    return std::fabs(actual - expected) <= tolerance;
}

void expect(bool condition, const char* expression, const char* file, int line) {
    if (condition) {
        return;
    }

    ++failure_count;
    std::cerr << "[FAIL] " << current_test << ": " << expression
              << " (" << file << ":" << line << ")\n";
}

#define EXPECT(condition) expect((condition), #condition, __FILE__, __LINE__)

void runTest(const char* name, const std::function<void()>& test) {
    current_test = name;
    int failures_before = failure_count;
    ++test_count;

    try {
        test();
    }
    catch (const std::exception& error) {
        ++failure_count;
        std::cerr << "[FAIL] " << current_test << ": unexpected exception: "
                  << error.what() << "\n";
    }
    catch (...) {
        ++failure_count;
        std::cerr << "[FAIL] " << current_test << ": unexpected non-standard exception\n";
    }

    if (failure_count == failures_before) {
        std::cout << "[PASS] " << current_test << "\n";
    }
}

Span makeSpan(float entry, float exit, uint32_t entry_id, uint32_t exit_id,
    const Vec3& entry_normal = Vec3(1, 0, 0),
    const Vec3& exit_normal = Vec3(-1, 0, 0)) {
    Span span;
    span.t_entry = entry;
    span.t_exit = exit;
    span.entry_hit = Hit(entry_normal, entry_id);
    span.exit_hit = Hit(exit_normal, exit_id);
    return span;
}

void testPrimitiveIntersections() {
    Span spans[1];
    uint32_t count = 0;

    float sphere_data[MAX_SHAPE_DATA_SIZE] = { 0, 0, 0, 1 };
    Sphere sphere(sphere_data);
    sphere.node_id = 7;
    sphere.getSpans(Ray(Vec3(0, 0, 3), Vec3(0, 0, -1)), spans, count);
    EXPECT(count == 1);
    EXPECT(nearlyEqual(spans[0].t_entry, 2.0f));
    EXPECT(nearlyEqual(spans[0].t_exit, 4.0f));
    EXPECT(spans[0].entry_hit.node_id == 7);
    EXPECT(nearlyEqual(spans[0].entry_hit.normal.z, 1.0f));

    float cuboid_data[MAX_SHAPE_DATA_SIZE] = { -1, -1, -1, 2, 2, 2 };
    Cuboid cuboid(cuboid_data);
    cuboid.node_id = 8;
    cuboid.getSpans(Ray(Vec3(0, 0, 3), Vec3(0, 0, -1)), spans, count);
    EXPECT(count == 1);
    EXPECT(nearlyEqual(spans[0].t_entry, 2.0f));
    EXPECT(nearlyEqual(spans[0].t_exit, 4.0f));
    EXPECT(spans[0].entry_hit.node_id == 8);

    float cylinder_data[MAX_SHAPE_DATA_SIZE] = { 0, 0, 0, 1, 2 };
    Cylinder cylinder(cylinder_data);
    cylinder.node_id = 9;
    cylinder.getSpans(Ray(Vec3(0, 1, 3), Vec3(0, 0, -1)), spans, count);
    EXPECT(count == 1);
    EXPECT(nearlyEqual(spans[0].t_entry, 2.0f));
    EXPECT(nearlyEqual(spans[0].t_exit, 4.0f));
    EXPECT(spans[0].entry_hit.node_id == 9);

    float cone_data[MAX_SHAPE_DATA_SIZE] = { 0, 0, 0, 1, 2 };
    Cone cone(cone_data);
    cone.node_id = 10;
    cone.getSpans(Ray(Vec3(0, 1, 3), Vec3(0, 0, -1)), spans, count);
    EXPECT(count == 1);
    EXPECT(nearlyEqual(spans[0].t_entry, 2.5f));
    EXPECT(nearlyEqual(spans[0].t_exit, 3.5f));
    EXPECT(spans[0].entry_hit.node_id == 10);
}

void testUnionSpans() {
    Span left_data[] = { makeSpan(1.0f, 3.0f, 10, 11) };
    Span right_data[] = { makeSpan(2.0f, 4.0f, 20, 21) };
    Span result_data[2];

    StridedSpan left(left_data);
    StridedSpan right(right_data);
    StridedSpan result(result_data);
    uint32_t result_count = 0;

    unionSpans(left, 1, right, 1, result, result_count);

    EXPECT(result_count == 1);
    EXPECT(nearlyEqual(result_data[0].t_entry, 1.0f));
    EXPECT(nearlyEqual(result_data[0].t_exit, 4.0f));
    EXPECT(result_data[0].entry_hit.node_id == 10);
    EXPECT(result_data[0].exit_hit.node_id == 21);
}

void testIntersectionSpans() {
    Span left_data[] = { makeSpan(1.0f, 3.0f, 10, 11) };
    Span right_data[] = { makeSpan(2.0f, 4.0f, 20, 21) };
    Span result_data[2];

    StridedSpan left(left_data);
    StridedSpan right(right_data);
    StridedSpan result(result_data);
    uint32_t result_count = 0;

    intersectionSpans(left, 1, right, 1, result, result_count);

    EXPECT(result_count == 1);
    EXPECT(nearlyEqual(result_data[0].t_entry, 2.0f));
    EXPECT(nearlyEqual(result_data[0].t_exit, 3.0f));
    EXPECT(result_data[0].entry_hit.node_id == 20);
    EXPECT(result_data[0].exit_hit.node_id == 11);
}

void testDifferenceSpans() {
    Span left_data[] = { makeSpan(1.0f, 5.0f, 10, 11) };
    Span right_data[] = {
        makeSpan(2.0f, 3.0f, 20, 21, Vec3(0, 1, 0), Vec3(0, -1, 0)),
        makeSpan(4.0f, 6.0f, 30, 31, Vec3(0, 0, 1), Vec3(0, 0, -1))
    };
    Span result_data[3];

    StridedSpan left(left_data);
    StridedSpan right(right_data);
    StridedSpan result(result_data);
    uint32_t result_count = 0;

    differenceSpans(left, 1, right, 2, result, result_count);

    EXPECT(result_count == 2);
    EXPECT(nearlyEqual(result_data[0].t_entry, 1.0f));
    EXPECT(nearlyEqual(result_data[0].t_exit, 2.0f));
    EXPECT(result_data[0].exit_hit.node_id == 20);
    EXPECT(nearlyEqual(result_data[0].exit_hit.normal.y, -1.0f));

    EXPECT(nearlyEqual(result_data[1].t_entry, 3.0f));
    EXPECT(nearlyEqual(result_data[1].t_exit, 4.0f));
    EXPECT(result_data[1].entry_hit.node_id == 21);
    EXPECT(nearlyEqual(result_data[1].entry_hit.normal.y, 1.0f));
    EXPECT(result_data[1].exit_hit.node_id == 30);
    EXPECT(nearlyEqual(result_data[1].exit_hit.normal.z, -1.0f));
}

void testSceneParserAndSizing() {
    std::string single_path = (repository_root / "single_sphere.txt").string();
    FlatCSGTree single = loadFromFile(single_path.c_str());

    EXPECT(single.num_nodes == 1);
    EXPECT(single.nodes[0].shape_type == ShapeType::Sphere);
    EXPECT(nearlyEqual(single.data[2], -5.0f));
    EXPECT(nearlyEqual(single.data[3], 1.0f));
    EXPECT(nearlyEqual(single.red[0], 1.0f));
    EXPECT(single.post_order_indexes[0] == 0);
    EXPECT(computeMaxDepth(single) == 1);
    EXPECT(computeTotalSpanUsage(single) == 1);
    freeHostTree(single);

    std::string union_path = (repository_root / "union_spheres.txt").string();
    FlatCSGTree combined = loadFromFile(union_path.c_str());

    EXPECT(combined.num_nodes == 3);
    EXPECT(combined.nodes[0].shape_type == ShapeType::TreeNode);
    EXPECT(combined.nodes[0].op == CSGOp::UNION);
    EXPECT(combined.left_indexes[0] == 1);
    EXPECT(combined.right_indexes[0] == 2);
    EXPECT(combined.post_order_indexes[0] == 1);
    EXPECT(combined.post_order_indexes[1] == 2);
    EXPECT(combined.post_order_indexes[2] == 0);
    EXPECT(computeMaxDepth(combined) == 2);
    EXPECT(computeTotalSpanUsage(combined) == 4);
    freeHostTree(combined);
}

void testParserRejectsUnknownOperation() {
    std::string invalid_path =
        (repository_root / "tests" / "fixtures" / "invalid_operation.txt").string();

    bool threw = false;
    try {
        FlatCSGTree tree = loadFromFile(invalid_path.c_str());
        freeHostTree(tree);
    }
    catch (const std::runtime_error&) {
        threw = true;
    }

    EXPECT(threw);
}

struct SingleSphereFixture {
    FlatCSGNodeInfo nodes[1];
    int32_t primitive_indexes[1];
    float data[MAX_SHAPE_DATA_SIZE];
    float red[1];
    float green[1];
    float blue[1];
    float diffuse[1];
    float specular[1];
    float shininess[1];
    uint32_t left_indexes[1];
    uint32_t right_indexes[1];
    uint32_t post_order[1];
    FlatCSGTree tree;

    SingleSphereFixture()
        : data{}, tree{} {
        nodes[0].op = CSGOp::UNION;
        nodes[0].shape_type = ShapeType::Sphere;
        primitive_indexes[0] = 0;
        data[0] = 0.0f;
        data[1] = 0.0f;
        data[2] = -5.0f;
        data[3] = 1.0f;
        red[0] = 1.0f;
        green[0] = 0.0f;
        blue[0] = 0.0f;
        diffuse[0] = 0.0f;
        specular[0] = 0.0f;
        shininess[0] = 1.0f;
        left_indexes[0] = 0;
        right_indexes[0] = 0;
        post_order[0] = 0;

        tree.num_nodes = 1;
        tree.num_primitives = 1;
        tree.nodes = nodes;
        tree.primitive_idx = primitive_indexes;
        tree.data = data;
        tree.red = red;
        tree.green = green;
        tree.blue = blue;
        tree.diffuse_coeff = diffuse;
        tree.specular_coeff = specular;
        tree.shininess = shininess;
        tree.left_indexes = left_indexes;
        tree.right_indexes = right_indexes;
        tree.post_order_indexes = post_order;
        tree.max_pool_size = 1;
        tree.max_stack_depth = 1;
    }
};

void testTreeEvaluationAndShading() {
    SingleSphereFixture fixture;
    Span pool_data[1];
    StackEntry stack_data[1];
    StridedSpan pool(pool_data);
    StridedStack stack(stack_data);

    size_t start_index = 99;
    uint32_t count = 99;
    Ray hit_ray(Vec3(0, 0, 0), Vec3(0, 0, -1));

    getSpans(hit_ray, &start_index, &count, fixture.tree, 0, pool, stack);
    EXPECT(start_index == 0);
    EXPECT(count == 1);
    EXPECT(nearlyEqual(pool_data[0].t_entry, 4.0f));
    EXPECT(nearlyEqual(pool_data[0].t_exit, 6.0f));

    Color hit_color = trace(
        hit_ray, Light(Vec3(0, 0, 1)), fixture.tree, pool, stack);
    EXPECT(nearlyEqual(hit_color.r, 0.2f));
    EXPECT(nearlyEqual(hit_color.g, 0.0f));
    EXPECT(nearlyEqual(hit_color.b, 0.0f));

    Color miss_color = trace(
        Ray(Vec3(0, 0, 0), Vec3(0, 1, 0)),
        Light(Vec3(0, 0, 1)), fixture.tree, pool, stack);
    EXPECT(nearlyEqual(miss_color.r, 0.0f));
    EXPECT(nearlyEqual(miss_color.g, 0.0f));
    EXPECT(nearlyEqual(miss_color.b, 0.0f));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: CSGRayCastTests <repository_root>\n";
        return 2;
    }

    repository_root = std::filesystem::path(argv[1]);

    runTest("primitive intersections", testPrimitiveIntersections);
    runTest("span union", testUnionSpans);
    runTest("span intersection", testIntersectionSpans);
    runTest("span difference", testDifferenceSpans);
    runTest("scene parser and sizing", testSceneParserAndSizing);
    runTest("invalid operation rejection", testParserRejectsUnknownOperation);
    runTest("tree evaluation and shading", testTreeEvaluationAndShading);

    if (failure_count != 0) {
        std::cerr << failure_count << " assertion(s) failed across "
                  << test_count << " test case(s).\n";
        return 1;
    }

    std::cout << "All " << test_count << " test cases passed.\n";
    return 0;
}
