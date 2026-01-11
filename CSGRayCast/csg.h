#pragma once





enum class CSGOp { UNION, INTERSECTION, DIFFERENCE };

enum class ShapeType { TreeNode, Sphere };

struct FlatCSGNodeInfo {
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
};

FlatCSGTree loadFromFile(const char* filename);







