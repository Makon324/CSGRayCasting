#pragma once

#include "shape.h"
#include <memory>


enum class CSGOp { UNION, INTERSECTION, DIFFERENCE };

class CSGNode : public Shape {
public:
    CSGOp op;
    Shape* shape;
    CSGNode* left;
    CSGNode* right;
public:
    CSGNode(CSGOp o, Shape* s, CSGNode* l, CSGNode* r) : op(o), shape(s), left(l), right(r) {}
    ~CSGNode() { delete left; delete right; delete shape; }  // Assumes ownership
    std::vector<Span> getSpans(const Ray& ray) const override;
};

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

    float* specular_coeff;
    float* shininess;

    size_t* left_indexes;
    size_t* right_indexes;
};

