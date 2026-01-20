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
    size_t num_primitives; // Count of actual shapes

    FlatCSGNodeInfo* nodes;

    int32_t* primitive_idx; // -1 for operators, 0..N for leaves

    // Arrays for primitive shape data
	float* data;  // sized to [num_primitives * MAX_SHAPE_DATA_SIZE]
    float* red;
    float* green;
    float* blue;
    float* diffuse_coeff;
    float* specular_coeff;
    float* shininess;

    // Topology data
    uint32_t* left_indexes;
    uint32_t* right_indexes;
    uint32_t* post_order_indexes;

    uint32_t max_pool_size;
    uint32_t max_stack_depth;
};

FlatCSGTree loadFromFile(const char* filename);

uint32_t computeMaxDepth(const FlatCSGTree& tree, uint32_t node_idx = 0);

uint32_t computeTotalSpanUsage(const FlatCSGTree& tree);



