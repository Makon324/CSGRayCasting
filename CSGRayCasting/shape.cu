#include <cuda_runtime.h>
#include <device_launch_parameters.h>   // helps with <<< >>> syntax highlighting & IntelliSense

#include <cstddef>      // size_t
#include <cmath>        // sqrtf, fmaxf, powf, ...

#include "csg.h"
#include "shape.h"
#include "scene.h"

#ifdef __INTELLISENSE__
void __syncthreads();
void __syncwarp();
#endif






constexpr int MAX_SPANS = 128;
constexpr int MAX_EVENTS = MAX_SPANS * 2;
constexpr int MAX_TREE_DEPTH = 32;

// Event struct for combining (make __device__ compatible)
struct Event {
    float t;
    bool is_entry;
    bool is_left;
    Hit hit;  // Copy hit instead of pointer for simplicity
    __device__ bool operator<(const Event& other) const {
        if (t != other.t) return t < other.t;
        return is_entry > other.is_entry;
    }
};

__host__ __device__ Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - normal * (2.0f * incident.dot(normal));
};

__device__ void warpBitonicSort(Event* events, int num_events) {
    if (num_events > 32) {
        // Serial bubble sort fallback
        for (int i = 0; i < num_events - 1; ++i) {
            for (int j = 0; j < num_events - i - 1; ++j) {
                if (events[j].t > events[j + 1].t || (events[j].t == events[j + 1].t && events[j].is_entry < events[j + 1].is_entry)) {
                    Event temp = events[j];
                    events[j] = events[j + 1];
                    events[j + 1] = temp;
                }
            }
        }
        return;
    }

    unsigned int lane = threadIdx.x % 32;
    Event temp;
    for (int k = 2; k <= 32; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = lane ^ j;
            if (ixj > lane) {
                if ((lane & k) == 0) {
                    if (events[lane].t > events[ixj].t || (events[lane].t == events[ixj].t && events[lane].is_entry < events[ixj].is_entry)) {
                        temp = events[lane];
                        events[lane] = events[ixj];
                        events[ixj] = temp;
                    }
                }
                else {
                    if (events[lane].t < events[ixj].t || (events[lane].t == events[ixj].t && events[lane].is_entry > events[ixj].is_entry)) {
                        temp = events[lane];
                        events[lane] = events[ixj];
                        events[ixj] = temp;
                    }
                }
            }
            __syncwarp();
        }
    }
}

__device__ void getSpansSphere(const float* sphere_data, const float red, const float green, const float blue,
    const float specular_coeff, const float shininess,
    const Ray& ray, Span* out_spans, int& num_out) {
    num_out = 0;
    Vec3 center(sphere_data[0], sphere_data[1], sphere_data[2]);
    float radius = sphere_data[3];

    Vec3 oc = ray.origin - center;
    float a = ray.dir.dot(ray.dir);
    float b = 2.0f * oc.dot(ray.dir);
    float c = oc.dot(oc) - radius * radius;

    float disc = b * b - 4.0f * a * c;
    if (disc < -1e-6f) return;
    if (disc < 0.0f) disc = 0.0f;

    float sqrt_disc = sqrtf(disc);
    float t1 = (-b - sqrt_disc) / (2.0f * a);
    float t2 = (-b + sqrt_disc) / (2.0f * a);
    if (t1 > t2) {
        float temp = t1; t1 = t2; t2 = temp;  // Swap without std::swap
    }
    if (t2 < 0.0f) return;
    if (t1 < 0.0f) t1 = 0.0f;

    Material mat{ Color(red, green, blue), specular_coeff, shininess };

    Vec3 entry_point = ray.at(t1);
    Vec3 entry_normal = (entry_point - center).normalize();

    Vec3 exit_point = ray.at(t2);
    Vec3 exit_normal = (exit_point - center).normalize();

    out_spans[0].t_entry = t1;
    out_spans[0].entry_hit = Hit(entry_normal, mat);
    out_spans[0].t_exit = t2;
    out_spans[0].exit_hit = Hit(exit_normal, mat);
    num_out = 1;
}

__device__ void getSpansShape(const FlatCSGNodeInfo& node, const float* shape_data_start,
    const float red, const float green, const float blue,
    const float specular_coeff, const float shininess,
    const Ray& ray, Span* out_spans, int& num_out) {
    num_out = 0;
    if (node.shape_type == ShapeType::Sphere) {
        getSpansSphere(shape_data_start, red, green, blue, specular_coeff, shininess, ray, out_spans, num_out);
    }
    // Else: other shapes in future
}

__device__ void combineSpansDevice(const Span* left_spans, int num_left,
    const Span* right_spans, int num_right,
    CSGOp op, Span* result_spans, int& num_result) {
    num_result = 0;
    if (num_left + num_right == 0) return;

    Event events[MAX_EVENTS];
    int num_events = 0;

    for (int i = 0; i < num_left; ++i) {
        events[num_events++] = { left_spans[i].t_entry, true, true, left_spans[i].entry_hit };
        events[num_events++] = { left_spans[i].t_exit, false, true, left_spans[i].exit_hit };
    }
    for (int i = 0; i < num_right; ++i) {
        events[num_events++] = { right_spans[i].t_entry, true, false, right_spans[i].entry_hit };
        events[num_events++] = { right_spans[i].t_exit, false, false, right_spans[i].exit_hit };
    }

    warpBitonicSort(events, num_events);

    int count_left = 0, count_right = 0;
    float start_t = -1.0f;
    Hit start_hit;
    bool prev_inside = false;

    for (int e_idx = 0; e_idx < num_events; ++e_idx) {
        const Event& e = events[e_idx];
        if (e.is_left) {
            if (e.is_entry) ++count_left;
            else --count_left;
        }
        else {
            if (e.is_entry) ++count_right;
            else --count_right;
        }

        bool curr_inside = false;
        switch (op) {
        case CSGOp::UNION:
            curr_inside = (count_left > 0 || count_right > 0);
            break;
        case CSGOp::INTERSECTION:
            curr_inside = (count_left > 0 && count_right > 0);
            break;
        case CSGOp::DIFFERENCE:
            curr_inside = (count_left > 0 && count_right == 0);
            break;
        }

        if (!prev_inside && curr_inside) {
            start_t = e.t;
            start_hit = e.hit;
            if (op == CSGOp::DIFFERENCE && !e.is_left) {
                start_hit.normal = -start_hit.normal;  // Flip
            }
        }
        else if (prev_inside && !curr_inside) {
            Span s;
            s.t_entry = start_t;
            s.entry_hit = start_hit;
            s.t_exit = e.t;
            s.exit_hit = e.hit;
            if (op == CSGOp::DIFFERENCE && !e.is_left) {
                s.exit_hit.normal = -s.exit_hit.normal;  // Flip
            }
            if (num_result < MAX_SPANS) {
                result_spans[num_result++] = s;
            }  // Else: overflow, truncate (rare)
            start_t = -1.0f;
        }
        prev_inside = curr_inside;
    }
}

__device__ void getTreeSpans(const FlatCSGTree& tree, const Ray& ray, Span* out_spans, int& num_out) {
    num_out = 0;
    if (tree.num_nodes == 0) return;

    size_t root_idx = tree.num_nodes - 1;  // Root is last in post-order flattening

    struct StackEntry {
        size_t idx;
        bool processed;
    };
    StackEntry stack[MAX_TREE_DEPTH];
    int top = 0;
    stack[top++] = { root_idx, false };

    struct SpanEntry {
        Span spans[MAX_SPANS];
        int num_spans;
    };
    SpanEntry value_stack[MAX_TREE_DEPTH];
    int value_top = 0;

    while (top > 0) {
        StackEntry& curr = stack[top - 1];

        if (curr.processed) {
            // Process node
            --top;
            size_t idx = curr.idx;
            const FlatCSGNodeInfo& info = tree.nodes[idx];
            bool is_leaf = (info.shape_type != ShapeType::TreeNode);

            Span local_spans[MAX_SPANS];
            int local_num = 0;

            if (is_leaf) {
                const float* data_start = &tree.data[idx * MAX_SHAPE_DATA_SIZE];
                getSpansShape(info, data_start, tree.red[idx], tree.green[idx], tree.blue[idx],
                    tree.specular_coeff[idx], tree.shininess[idx], ray, local_spans, local_num);
            }
            else {
                // Combine children (pop right then left)
                SpanEntry right = value_stack[--value_top];
                SpanEntry left = value_stack[--value_top];
                combineSpansDevice(left.spans, left.num_spans, right.spans, right.num_spans,
                    info.op, local_spans, local_num);
            }

            // Push to value stack
            value_stack[value_top].num_spans = local_num;
            for (int i = 0; i < local_num; ++i) {
                value_stack[value_top].spans[i] = local_spans[i];
            }
            ++value_top;
        }
        else {
            // Mark processed and push children (right then left for post-order)
            curr.processed = true;
            size_t right = tree.right_indexes[curr.idx];
            if (right != size_t(-1)) {
                stack[top++] = { right, false };
            }
            size_t left = tree.left_indexes[curr.idx];
            if (left != size_t(-1)) {
                stack[top++] = { left, false };
            }
        }
    }

    // Root result at top (value_top - 1)
    if (value_top > 0) {
        SpanEntry top_entry = value_stack[value_top - 1];
        num_out = top_entry.num_spans;
        for (int i = 0; i < num_out; ++i) {
            out_spans[i] = top_entry.spans[i];
        }
    }
}

__device__ Color phongShadeDevice(const Vec3& point, const Vec3& normal, const Material& mat,
    const Vec3& view_dir, const Light& light, const FlatCSGTree& tree) {
    Vec3 l_dir = light.direction;

    // Shadow ray
    Ray shadow_ray(point + normal * EPS, l_dir);
    Span shadow_spans[MAX_SPANS];
    int num_shadow = 0;
    getTreeSpans(tree, shadow_ray, shadow_spans, num_shadow);
    bool in_shadow = false;
    for (int i = 0; i < num_shadow; ++i) {
        if (shadow_spans[i].t_entry > EPS) {
            in_shadow = true;
            break;
        }
    }

    // Ambient
    Color ambient = mat.color * 0.1f;

    if (in_shadow) return ambient;

    // Diffuse
    float diff = fmaxf(0.0f, normal.dot(l_dir));
    Color diffuse = mat.color * diff;

    // Specular
    Vec3 r = reflect(-l_dir, normal);
    float spec = powf(fmaxf(0.0f, view_dir.dot(r)), mat.shininess) * mat.specular_coeff;
    Color specular = Color(1.0f, 1.0f, 1.0f) * spec;

    return ambient + diffuse + specular;
}

__device__ Color traceDevice(const Ray& ray, const Light& light, const FlatCSGTree& tree) {
    Span spans[MAX_SPANS];
    int num_spans = 0;
    getTreeSpans(tree, ray, spans, num_spans);
    if (num_spans == 0) return Color(0, 0, 0);

    float min_t = INF;
    Hit closest_hit;
    for (int i = 0; i < num_spans; ++i) {
        if (spans[i].t_entry > EPS && spans[i].t_entry < min_t) {
            min_t = spans[i].t_entry;
            closest_hit = spans[i].entry_hit;
        }
    }
    if (min_t == INF) return Color(0, 0, 0);

    Vec3 point = ray.at(min_t);
    Vec3 normal = closest_hit.normal;
    Vec3 view_dir = -ray.dir;

    return phongShadeDevice(point, normal, closest_hit.mat, view_dir, light, tree);
}

// Add kernel in shape.cu or scene.cu
__global__ void renderKernel(unsigned char* d_output, int width, int height, const Camera cam,
    const Light light, const FlatCSGTree d_tree) {
    extern __shared__ char sh_mem[];

    size_t num_nodes = d_tree.num_nodes;
    if (num_nodes == 0) return;

    // Partition shared memory
    size_t offset = 0;
    FlatCSGNodeInfo* sh_nodes = (FlatCSGNodeInfo*)&sh_mem[offset];
    offset += num_nodes * sizeof(FlatCSGNodeInfo);
    float* sh_data = (float*)&sh_mem[offset];
    offset += num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float);
    float* sh_red = (float*)&sh_mem[offset];
    offset += num_nodes * sizeof(float);
    float* sh_green = (float*)&sh_mem[offset];
    offset += num_nodes * sizeof(float);
    float* sh_blue = (float*)&sh_mem[offset];
    offset += num_nodes * sizeof(float);
    float* sh_specular_coeff = (float*)&sh_mem[offset];
    offset += num_nodes * sizeof(float);
    float* sh_shininess = (float*)&sh_mem[offset];
    offset += num_nodes * sizeof(float);
    size_t* sh_left_indexes = (size_t*)&sh_mem[offset];
    offset += num_nodes * sizeof(size_t);
    size_t* sh_right_indexes = (size_t*)&sh_mem[offset];

    // Cooperative copy from global to shared
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    for (size_t i = tid; i < num_nodes; i += stride) {
        sh_nodes[i] = d_tree.nodes[i];
        sh_red[i] = d_tree.red[i];
        sh_green[i] = d_tree.green[i];
        sh_blue[i] = d_tree.blue[i];
        sh_specular_coeff[i] = d_tree.specular_coeff[i];
        sh_shininess[i] = d_tree.shininess[i];
        sh_left_indexes[i] = d_tree.left_indexes[i];
        sh_right_indexes[i] = d_tree.right_indexes[i];
        for (int d = 0; d < MAX_SHAPE_DATA_SIZE; ++d) {
            sh_data[i * MAX_SHAPE_DATA_SIZE + d] = d_tree.data[i * MAX_SHAPE_DATA_SIZE + d];
        }
    }
    __syncthreads();

    // Rebuild shared tree
    FlatCSGTree sh_tree;
    sh_tree.num_nodes = num_nodes;
    sh_tree.nodes = sh_nodes;
    sh_tree.data = sh_data;
    sh_tree.red = sh_red;
    sh_tree.green = sh_green;
    sh_tree.blue = sh_blue;
    sh_tree.specular_coeff = sh_specular_coeff;
    sh_tree.shininess = sh_shininess;
    sh_tree.left_indexes = sh_left_indexes;
    sh_tree.right_indexes = sh_right_indexes;

    // Compute pixel
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    float s = (i + 0.5f) / width;
    float t = (j + 0.5f) / height;
    Ray r = cam.getRay(s, t);
    Color color = traceDevice(r, light, sh_tree);

    // Clamp and store (RGB uchar)
    size_t pixel_idx = (j * width + i) * 3;
    d_output[pixel_idx + 0] = (unsigned char)fminf(255, (int)(255.99f * fmaxf(0.0f, color.r)));
    d_output[pixel_idx + 1] = (unsigned char)fminf(255, (int)(255.99f * fmaxf(0.0f, color.g)));
    d_output[pixel_idx + 2] = (unsigned char)fminf(255, (int)(255.99f * fmaxf(0.0f, color.b)));
}


