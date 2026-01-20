#pragma once

#include <cstddef>
#include <cstdint>


enum class CSGOp : uint16_t { UNION, INTERSECTION, DIFFERENCE };

enum class ShapeType : uint16_t { TreeNode, Sphere, Cuboid, Cylinder, Cone };

struct alignas(4) FlatCSGNodeInfo {
    CSGOp op;
    ShapeType shape_type;
    int32_t primitive_idx; // -1 for operators, 0..N for leaves
};

struct FlatCSGTree {
    size_t num_nodes;
    size_t num_primitives; // NEW: Count of actual shapes

    FlatCSGNodeInfo* nodes;

    // These arrays are now sized to [num_primitives] instead of [num_nodes]
    float* data;
    float* red;
    float* green;
    float* blue;
    float* diffuse_coeff;
    float* specular_coeff;
    float* shininess;

    // Topology remains sized to [num_nodes]
    size_t* left_indexes;
    size_t* right_indexes;
    size_t* post_order_indexes;

    size_t max_pool_size;
    size_t max_stack_depth;
};

FlatCSGTree loadFromFile(const char* filename);

size_t computeMaxDepth(const FlatCSGTree& tree, size_t node_idx = 0);

size_t computeTotalSpanUsage(const FlatCSGTree& tree);



