#include <iostream>
#include <string>
#include "scene.h"

// Function to build a simple single red sphere scene
CSGNode* buildSingleSphere() {
    Material mat_red{ Color(1.0f, 0.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Sphere* s1 = new Sphere(Vec3(0, 0, -5), 1.0f, mat_red);
    CSGNode* leaf1 = new CSGNode(CSGOp::UNION, s1, nullptr, nullptr);
    return leaf1;  // Single sphere as root
}

// Function to demonstrate union: two overlapping spheres (red and green)
CSGNode* buildUnionSpheres() {
    Material mat_red{ Color(1.0f, 0.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Material mat_green{ Color(0.0f, 1.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Sphere* s1 = new Sphere(Vec3(0, 0, -5), 1.0f, mat_red);
    Sphere* s2 = new Sphere(Vec3(0.5f, 0, -5), 0.7f, mat_green);
    CSGNode* leaf1 = new CSGNode(CSGOp::UNION, s1, nullptr, nullptr);
    CSGNode* leaf2 = new CSGNode(CSGOp::UNION, s2, nullptr, nullptr);
    CSGNode* root = new CSGNode(CSGOp::UNION, nullptr, leaf1, leaf2);
    return root;
}

// Function to demonstrate difference: red sphere minus green sphere
CSGNode* buildDifferenceSpheres() {
    Material mat_red{ Color(1.0f, 0.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Material mat_green{ Color(0.0f, 1.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Sphere* s1 = new Sphere(Vec3(0, 0, -5), 1.0f, mat_red);
    Sphere* s2 = new Sphere(Vec3(0.5f, 0, -5), 0.7f, mat_green);
    CSGNode* leaf1 = new CSGNode(CSGOp::UNION, s1, nullptr, nullptr);
    CSGNode* leaf2 = new CSGNode(CSGOp::UNION, s2, nullptr, nullptr);
    CSGNode* root = new CSGNode(CSGOp::DIFFERENCE, nullptr, leaf1, leaf2);
    return root;
}

// Function to demonstrate intersection: overlapping region of red and blue spheres
CSGNode* buildIntersectionSpheres() {
    Material mat_red{ Color(1.0f, 0.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Material mat_blue{ Color(0.0f, 0.0f, 1.0f), 0.5f, 32.0f, 0.3f };
    Sphere* s1 = new Sphere(Vec3(0, 0, -5), 1.0f, mat_red);
    Sphere* s2 = new Sphere(Vec3(0.5f, 0, -5), 1.0f, mat_blue);
    CSGNode* leaf1 = new CSGNode(CSGOp::UNION, s1, nullptr, nullptr);
    CSGNode* leaf2 = new CSGNode(CSGOp::UNION, s2, nullptr, nullptr);
    CSGNode* root = new CSGNode(CSGOp::INTERSECTION, nullptr, leaf1, leaf2);
    return root;
}

// Function to demonstrate reflections: a highly reflective sphere with another nearby
CSGNode* buildReflectiveScene() {
    Material mat_mirror{ Color(0.8f, 0.8f, 0.8f), 1.0f, 128.0f, 0.9f };  // High reflectivity
    Material mat_green{ Color(0.0f, 1.0f, 0.0f), 0.5f, 32.0f, 0.1f };   // Low reflectivity
    Sphere* s1 = new Sphere(Vec3(0, 0, -5), 1.0f, mat_mirror);
    Sphere* s2 = new Sphere(Vec3(2.0f, 0, -5), 0.8f, mat_green);
    CSGNode* leaf1 = new CSGNode(CSGOp::UNION, s1, nullptr, nullptr);
    CSGNode* leaf2 = new CSGNode(CSGOp::UNION, s2, nullptr, nullptr);
    CSGNode* root = new CSGNode(CSGOp::UNION, nullptr, leaf1, leaf2);
    return root;
}

// Function to build a super complicated scene: a large sphere with multiple holes (difference of union of small spheres),
// intersected with another shape, and unioned with reflective elements
CSGNode* buildComplexScene() {
    // Materials
    Material mat_red{ Color(1.0f, 0.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Material mat_green{ Color(0.0f, 1.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Material mat_blue{ Color(0.0f, 0.0f, 1.0f), 0.5f, 32.0f, 0.3f };
    Material mat_yellow{ Color(1.0f, 1.0f, 0.0f), 0.5f, 32.0f, 0.3f };
    Material mat_mirror{ Color(0.8f, 0.8f, 0.8f), 1.0f, 128.0f, 0.9f };

    // Large base sphere
    Sphere* large = new Sphere(Vec3(0, 0, -5), 2.0f, mat_red);
    CSGNode* large_leaf = new CSGNode(CSGOp::UNION, large, nullptr, nullptr);

    // Small hole spheres to subtract (like swiss cheese)
    Sphere* hole1 = new Sphere(Vec3(0.5f, 0.5f, -5), 0.5f, mat_green);
    Sphere* hole2 = new Sphere(Vec3(-0.5f, 0.5f, -5), 0.5f, mat_green);
    Sphere* hole3 = new Sphere(Vec3(0.5f, -0.5f, -5), 0.5f, mat_green);
    Sphere* hole4 = new Sphere(Vec3(-0.5f, -0.5f, -5), 0.5f, mat_green);
    Sphere* hole5 = new Sphere(Vec3(0, 0, -4.5f), 0.4f, mat_green);
    Sphere* hole6 = new Sphere(Vec3(0, 0, -5.5f), 0.4f, mat_green);

    // Build union of holes as a binary tree
    CSGNode* hole1_leaf = new CSGNode(CSGOp::UNION, hole1, nullptr, nullptr);
    CSGNode* hole2_leaf = new CSGNode(CSGOp::UNION, hole2, nullptr, nullptr);
    CSGNode* union12 = new CSGNode(CSGOp::UNION, nullptr, hole1_leaf, hole2_leaf);

    CSGNode* hole3_leaf = new CSGNode(CSGOp::UNION, hole3, nullptr, nullptr);
    CSGNode* hole4_leaf = new CSGNode(CSGOp::UNION, hole4, nullptr, nullptr);
    CSGNode* union34 = new CSGNode(CSGOp::UNION, nullptr, hole3_leaf, hole4_leaf);

    CSGNode* hole5_leaf = new CSGNode(CSGOp::UNION, hole5, nullptr, nullptr);
    CSGNode* hole6_leaf = new CSGNode(CSGOp::UNION, hole6, nullptr, nullptr);
    CSGNode* union56 = new CSGNode(CSGOp::UNION, nullptr, hole5_leaf, hole6_leaf);

    // Union of first four
    CSGNode* union1234 = new CSGNode(CSGOp::UNION, nullptr, union12, union34);

    // Union all holes
    CSGNode* all_holes = new CSGNode(CSGOp::UNION, nullptr, union1234, union56);

    // Subtract holes from large sphere (difference)
    CSGNode* swiss_cheese = new CSGNode(CSGOp::DIFFERENCE, nullptr, large_leaf, all_holes);

    // Another shape to intersect with: a blue sphere
    Sphere* intersect_blue = new Sphere(Vec3(0, 1.0f, -5), 1.5f, mat_blue);
    CSGNode* blue_leaf = new CSGNode(CSGOp::UNION, intersect_blue, nullptr, nullptr);

    // Intersect the swiss cheese with the blue sphere
    CSGNode* intersected = new CSGNode(CSGOp::INTERSECTION, nullptr, swiss_cheese, blue_leaf);

    // Add a reflective yellow sphere nearby
    Sphere* reflective_yellow = new Sphere(Vec3(3.0f, 0, -5), 1.0f, mat_mirror);
    CSGNode* yellow_leaf = new CSGNode(CSGOp::UNION, reflective_yellow, nullptr, nullptr);

    // Union the intersected shape with the reflective sphere
    CSGNode* root = new CSGNode(CSGOp::UNION, nullptr, intersected, yellow_leaf);

    return root;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [cpu|gpu]\n";
        return 1;
    }
    std::string mode = argv[1];

    // Build the scene - swap the function call here to change scenes
    // CSGNode* root = buildSingleSphere();
    // CSGNode* root = buildUnionSpheres();
    // CSGNode* root = buildDifferenceSpheres();
    // CSGNode* root = buildIntersectionSpheres();
    // CSGNode* root = buildReflectiveScene();
    CSGNode* root = buildComplexScene();  // Default to the super complicated scene

    Light light(Vec3(5, 5, 0));
    Camera cam(Vec3(0, 0, 0), Vec3(0, 0, -5), Vec3(0, 1, 0), 60.0f, 800, 600);
    Scene scene(root, light, cam);

    if (mode == "cpu") {
        scene.render("output.ppm");
        std::cout << "Rendered to output.ppm using CPU.\n";
    }
    else if (mode == "gpu") {
        // Placeholder for GPU (e.g., call a GPU renderer function when implemented)
        std::cout << "GPU rendering not implemented yet.\n";
    }
    else {
        std::cerr << "Invalid mode: use 'cpu' or 'gpu'.\n";
    }

    delete root;  // Cleans up nodes and shapes (note: add 'delete shape;' in ~CSGNode() to fully clean shapes)
    return 0;
}