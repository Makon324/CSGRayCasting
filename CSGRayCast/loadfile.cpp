#include "csg.h"
#include "shape.h"

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

static size_t parseSubtree(std::istream& in,
    std::vector<FlatCSGNodeInfo>& nodes,
    std::vector<float>& shape_data,
    std::vector<float>& reds,
    std::vector<float>& greens,
    std::vector<float>& blues,
    std::vector<float>& diffs,
    std::vector<float>& specs,
    std::vector<float>& shins,
    std::vector<size_t>& lefts,
    std::vector<size_t>& rights) {
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Unexpected end of file");
    }

    // Trim leading spaces and count indent
    size_t indent = 0;
    while (indent < line.size() && line[indent] == ' ') {
        ++indent;
    }
    std::string content = line.substr(indent);
    if (content.empty()) {
        // Skip empty lines
        return parseSubtree(in, nodes, shape_data, reds, greens, blues, diffs, specs, shins, lefts, rights);
    }

    std::stringstream ss(content);
    std::string type;
    ss >> type;

    size_t this_index = nodes.size();
    nodes.emplace_back();
    shape_data.resize(nodes.size() * MAX_SHAPE_DATA_SIZE);
    reds.resize(nodes.size());
    greens.resize(nodes.size());
    blues.resize(nodes.size());
    diffs.resize(nodes.size());
    specs.resize(nodes.size());
    shins.resize(nodes.size());
    lefts.resize(nodes.size());
    rights.resize(nodes.size());

    std::vector<float> this_data(MAX_SHAPE_DATA_SIZE, 0.0f);
    float r = 0.0f, g = 0.0f, b = 0.0f, diff = 1.0f, spec = 0.0f, shin = 0.0f;
    size_t left = 0, right = 0;  // 0 indicates no child

    if (type == "sphere") {
        nodes[this_index].shape_type = ShapeType::Sphere;
        // op is irrelevant for leaves

        float x, y, z, rad, cr, cg, cb, dc, sc, sh;
        if (!(ss >> x >> y >> z >> rad >> cr >> cg >> cb >> dc >> sc >> sh)) {
            throw std::runtime_error("Parse error in sphere data");
        }

        this_data[0] = x;
        this_data[1] = y;
        this_data[2] = z;
        this_data[3] = rad;

        r = cr;
        g = cg;
        b = cb;
        diff = dc;
        spec = sc;
        shin = sh;
    }
    else {
        nodes[this_index].shape_type = ShapeType::TreeNode;
        if (type == "union") {
            nodes[this_index].op = CSGOp::UNION;
        }
        else if (type == "intersection") {
            nodes[this_index].op = CSGOp::INTERSECTION;
        }
        else if (type == "difference") {
            nodes[this_index].op = CSGOp::DIFFERENCE;
        }
        else {
            throw std::runtime_error("Unknown CSG operation: " + type);
        }

        // Parse left child
        left = parseSubtree(in, nodes, shape_data, reds, greens, blues, diffs, specs, shins, lefts, rights);
        // Parse right child
        right = parseSubtree(in, nodes, shape_data, reds, greens, blues, diffs, specs, shins, lefts, rights);
    }

    for (size_t i = 0; i < MAX_SHAPE_DATA_SIZE; ++i) {
        shape_data[this_index * MAX_SHAPE_DATA_SIZE + i] = this_data[i];
    }
    reds[this_index] = r;
    greens[this_index] = g;
    blues[this_index] = b;
    diffs[this_index] = diff;
    specs[this_index] = spec;
    shins[this_index] = shin;
    lefts[this_index] = left;
    rights[this_index] = right;

    return this_index;
}

void buildPostOrder(const FlatCSGTree& tree, size_t idx, std::vector<size_t>& post_order) {
    if (tree.nodes[idx].shape_type == ShapeType::TreeNode) {
        buildPostOrder(tree, tree.left_indexes[idx], post_order);
        buildPostOrder(tree, tree.right_indexes[idx], post_order);
    }
    post_order.push_back(idx);
}

FlatCSGTree loadFromFile(const char* filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + std::string(filename));
    }

    std::vector<FlatCSGNodeInfo> nodes;
    std::vector<float> shape_data;
    std::vector<float> reds, greens, blues, diffs, specs, shins;
    std::vector<size_t> lefts, rights;

    try {
        parseSubtree(in, nodes, shape_data, reds, greens, blues, diffs, specs, shins, lefts, rights);
    }
    catch (const std::exception& e) {
        // Clean up any partial allocations if needed, but since we throw, user handles
        throw;
    }

    size_t num_nodes = nodes.size();
    if (num_nodes == 0) {
        return {};
    }

    FlatCSGTree tree;
    tree.num_nodes = num_nodes;

    tree.nodes = new FlatCSGNodeInfo[num_nodes];
    std::memcpy(tree.nodes, nodes.data(), num_nodes * sizeof(FlatCSGNodeInfo));

    tree.data = new float[shape_data.size()];
    std::memcpy(tree.data, shape_data.data(), shape_data.size() * sizeof(float));

    tree.red = new float[num_nodes];
    std::memcpy(tree.red, reds.data(), num_nodes * sizeof(float));

    tree.green = new float[num_nodes];
    std::memcpy(tree.green, greens.data(), num_nodes * sizeof(float));

    tree.blue = new float[num_nodes];
    std::memcpy(tree.blue, blues.data(), num_nodes * sizeof(float));

    tree.diffuse_coeff = new float[num_nodes];
    std::memcpy(tree.diffuse_coeff, diffs.data(), num_nodes * sizeof(float));

    tree.specular_coeff = new float[num_nodes];
    std::memcpy(tree.specular_coeff, specs.data(), num_nodes * sizeof(float));

    tree.shininess = new float[num_nodes];
    std::memcpy(tree.shininess, shins.data(), num_nodes * sizeof(float));

    tree.left_indexes = new size_t[num_nodes];
    std::memcpy(tree.left_indexes, lefts.data(), num_nodes * sizeof(size_t));

    tree.right_indexes = new size_t[num_nodes];
    std::memcpy(tree.right_indexes, rights.data(), num_nodes * sizeof(size_t));

    // Compute post-order traversal
    std::vector<size_t> post_order;
    buildPostOrder(tree, 0, post_order);
    tree.post_order_indexes = new size_t[num_nodes];
    std::memcpy(tree.post_order_indexes, post_order.data(), num_nodes * sizeof(size_t));

    return tree;
}