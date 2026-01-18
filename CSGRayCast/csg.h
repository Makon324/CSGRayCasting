#pragma once

#include <cstddef>
#include <cstdint>


enum class CSGOp : uint16_t { UNION, INTERSECTION, DIFFERENCE };

enum class ShapeType : uint16_t { TreeNode, Sphere, Cuboid, Cylinder, Cone };

struct alignas(4) FlatCSGNodeInfo {
    CSGOp op;
    ShapeType shape_type;
};

struct FlatCSGTree {
    size_t num_nodes;

    FlatCSGNodeInfo* nodes;

    float* data;  // shape data stored in a flat array (num_nodes * MAX_SHAPE_DATA_SIZE)

    float* red;
    float* green;
    float* blue;

    float* diffuse_coeff;
    float* specular_coeff;
    float* shininess;

    size_t* left_indexes;
    size_t* right_indexes;

    size_t* post_order_indexes;

    // Computed sizes for dynamic allocation
    size_t max_pool_size;
    size_t max_stack_depth;
};

FlatCSGTree loadFromFile(const char* filename);

size_t computeMaxDepth(const FlatCSGTree& tree, size_t node_idx = 0);





