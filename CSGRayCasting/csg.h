#pragma once

#include "shape.h"
#include <memory>


enum class CSGOp { UNION, INTERSECTION, DIFFERENCE };

class CSGNode : public Shape {
    CSGOp op;
    Shape* shape;
    CSGNode* left;
    CSGNode* right;
public:
    CSGNode(CSGOp o, Shape* s, CSGNode* l, CSGNode* r) : op(o), shape(s), left(l), right(r) {}
    ~CSGNode() { delete left; delete right; delete shape; }  // Assumes ownership
    std::vector<Span> getSpans(const Ray& ray) const override;
};

