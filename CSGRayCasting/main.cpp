#include <iostream>
#include <string>
#include <fstream>
#include "scene.h"

// Recursive function to parse a CSG node from the input stream
CSGNode* parseNode(std::istream& in) {
    std::string token;
    if (!(in >> token)) {
        std::cerr << "Error: Unexpected end of file while parsing node.\n";
        return nullptr;
    }

    if (token == "sphere") {
        Vec3 center;
        if (!(in >> center.x >> center.y >> center.z)) {
            std::cerr << "Error: Failed to parse sphere center.\n";
            return nullptr;
        }
        float radius;
        if (!(in >> radius)) {
            std::cerr << "Error: Failed to parse sphere radius.\n";
            return nullptr;
        }
        Color col;
        if (!(in >> col.r >> col.g >> col.b)) {
            std::cerr << "Error: Failed to parse sphere color.\n";
            return nullptr;
        }
        float specular_coeff, shininess;
        if (!(in >> specular_coeff >> shininess)) {
            std::cerr << "Error: Failed to parse sphere material properties.\n";
            return nullptr;
        }
        Material mat{ col, specular_coeff, shininess};
        Sphere* s = new Sphere(center, radius, mat);
        return new CSGNode(CSGOp::UNION, s, nullptr, nullptr);
    }

    else {
        CSGOp op;
        if (token == "union") {
            op = CSGOp::UNION;
        }
        else if (token == "intersection") {
            op = CSGOp::INTERSECTION;
        }
        else if (token == "difference") {
            op = CSGOp::DIFFERENCE;
        }
        else {
            std::cerr << "Error: Unknown token '" << token << "' (expected 'sphere', 'union', 'intersection', or 'difference').\n";
            return nullptr;
        }
        CSGNode* left = parseNode(in);
        if (!left) return nullptr;
        CSGNode* right = parseNode(in);
        if (!right) {
            delete left;
            return nullptr;
        }
        return new CSGNode(op, nullptr, left, right);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [cpu|gpu] [scene_file]\n";
        return 1;
    }
    std::string mode = argv[1];
    std::string filename = argv[2];

    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: Could not open scene file '" << filename << "'.\n";
        return 1;
    }

    CSGNode* root = parseNode(in);
    if (!root) {
        std::cerr << "Error: Failed to parse scene.\n";
        return 1;
    }

    Light light(Vec3(5, 5, 0));  // Interpreted as direction, will be normalized in constructor
    Camera cam(Vec3(0, 0, 0), Vec3(0, 0, -5), Vec3(0, 1, 0), 60.0f, 800, 600);
    Scene scene(root, light, cam);

    if (mode == "cpu") {
        scene.renderCPU("output.ppm");
        std::cout << "Rendered to output.ppm using CPU.\n";
    }
    else if (mode == "gpu") {
        scene.renderGPU("output.ppm");
        std::cout << "Rendered to output.ppm using GPU.\n";
    }
    else {
        std::cerr << "Invalid mode: use 'cpu' or 'gpu'.\n";
    }

    delete root;  // Cleans up nodes and shapes
    return 0;
}