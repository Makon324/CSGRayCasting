// Corrected tracer.cu with fixes for compilation errors and warnings

#include "shape.h"
#include "tracer.h"
#include "csg.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

// Define fixed maximum sizes for GPU local arrays to avoid VLAs
constexpr int MAX_POOL_SIZE = 2048;  // Adjust as needed based on expected tree size (e.g., for num_nodes ~30, this is plenty)
constexpr int MAX_STACK_DEPTH = 128; // Sufficient for deep trees

__host__ __device__ Vec3 reflect(const Vec3& I, const Vec3& N) {
    return I - N * 2.f * I.dot(N);
}

__host__ __device__ void unionSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count) {
    result_count = 0;
    if (left_count == 0 && right_count == 0) return;

    Span all[2 * MAX_SPANS];
    uint32_t all_count = 0;

    // Merge left and right assuming they are sorted by t_entry
    uint32_t li = 0, ri = 0;
    while (li < left_count && ri < right_count) {
        if (left[li].t_entry < right[ri].t_entry) {
            all[all_count++] = left[li++];
        }
        else {
            all[all_count++] = right[ri++];
        }
    }
    while (li < left_count) all[all_count++] = left[li++];
    while (ri < right_count) all[all_count++] = right[ri++];

    // Now merge overlapping intervals
    float current_start = all[0].t_entry;
    float current_end = all[0].t_exit;
    Hit entry_hit = all[0].entry_hit;
    Hit exit_hit = all[0].exit_hit;
    for (uint32_t i = 1; i < all_count; ++i) {
        if (all[i].t_entry <= current_end + 1e-6f) {  // Epsilon for floating-point precision
            if (all[i].t_exit > current_end) {
                current_end = all[i].t_exit;
                exit_hit = all[i].exit_hit;
            }
        }
        else {
            result[result_count].t_entry = current_start;
            result[result_count].entry_hit = entry_hit;
            result[result_count].t_exit = current_end;
            result[result_count].exit_hit = exit_hit;
            ++result_count;
            current_start = all[i].t_entry;
            current_end = all[i].t_exit;
            entry_hit = all[i].entry_hit;
            exit_hit = all[i].exit_hit;
        }
    }
    result[result_count].t_entry = current_start;
    result[result_count].entry_hit = entry_hit;
    result[result_count].t_exit = current_end;
    result[result_count].exit_hit = exit_hit;
    ++result_count;
}

__host__ __device__ void intersectionSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count) {
    result_count = 0;
    Span temp[MAX_SPANS];
    uint32_t temp_count = 0;
    for (uint32_t i = 0; i < left_count; ++i) {
        for (uint32_t j = 0; j < right_count; ++j) {
            float start = (left[i].t_entry > right[j].t_entry) ? left[i].t_entry : right[j].t_entry;
            float end = (left[i].t_exit < right[j].t_exit) ? left[i].t_exit : right[j].t_exit;
            if (start < end) {
                Hit e_hit = (left[i].t_entry > right[j].t_entry) ? left[i].entry_hit : right[j].entry_hit;
                Hit x_hit = (left[i].t_exit < right[j].t_exit) ? left[i].exit_hit : right[j].exit_hit;
                temp[temp_count].t_entry = start;
                temp[temp_count].entry_hit = e_hit;
                temp[temp_count].t_exit = end;
                temp[temp_count].exit_hit = x_hit;
                ++temp_count;
            }
        }
    }
    // Insertion sort temp by t_entry (better than bubble for small n)
    for (uint32_t i = 1; i < temp_count; ++i) {
        Span key = temp[i];
        uint32_t j = i;
        while (j > 0 && temp[j - 1].t_entry > key.t_entry) {
            temp[j] = temp[j - 1];
            --j;
        }
        temp[j] = key;
    }
    result_count = temp_count;
    for (uint32_t i = 0; i < temp_count; ++i) result[i] = temp[i];
}

__host__ __device__ void differenceSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count) {
    result_count = 0;
    Span temp[2 * MAX_SPANS];
    uint32_t temp_count = 0;
    for (uint32_t li = 0; li < left_count; ++li) {
        Span current = left[li];
        for (uint32_t ri = 0; ri < right_count; ++ri) {
            if (right[ri].t_exit <= current.t_entry || right[ri].t_entry >= current.t_exit) continue;
            if (right[ri].t_entry > current.t_entry) {
                Span new_span;
                new_span.t_entry = current.t_entry;
                new_span.entry_hit = current.entry_hit;
                new_span.t_exit = right[ri].t_entry;
                new_span.exit_hit = right[ri].entry_hit;
                new_span.exit_hit.normal = -new_span.exit_hit.normal;
                temp[temp_count++] = new_span;
            }
            current.t_entry = (current.t_entry > right[ri].t_exit) ? current.t_entry : right[ri].t_exit;
            current.entry_hit = right[ri].exit_hit;
            current.entry_hit.normal = -current.entry_hit.normal;
        }
        if (current.t_entry < current.t_exit) {
            temp[temp_count++] = current;
        }
    }
    // Insertion sort temp by t_entry
    for (uint32_t i = 1; i < temp_count; ++i) {
        Span key = temp[i];
        uint32_t j = i;
        while (j > 0 && temp[j - 1].t_entry > key.t_entry) {
            temp[j] = temp[j - 1];
            --j;
        }
        temp[j] = key;
    }
    result_count = temp_count;
    for (uint32_t i = 0; i < temp_count; ++i) result[i] = temp[i];
}

__host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count, const FlatCSGTree& tree, size_t node_idx) {
    count = 0;
    if (tree.num_nodes == 0) return;

    // New: Use tree's computed sizes for VLAs
    Span* pool = nullptr;
    int pool_ptr = 0;

    struct StackEntry {
        int start;
        uint32_t count;
    };
    StackEntry* stack = nullptr;
    int sp = 0;

#ifdef __CUDA_ARCH__  // Device (GPU) code: Use fixed-size arrays
    Span pool_array[MAX_POOL_SIZE];
    StackEntry stack_array[MAX_STACK_DEPTH];
    pool = pool_array;
    stack = stack_array;
    // Check if the tree fits within fixed limits
    if (tree.max_pool_size > MAX_POOL_SIZE || tree.max_stack_depth > MAX_STACK_DEPTH) {
        count = 0;  // Overflow: handle by skipping (render black or error color)
        return;
    }
#else  // Host (CPU) code: Use std::vector for dynamic sizing
    std::vector<Span> pool_vec(tree.max_pool_size);
    std::vector<StackEntry> stack_vec(tree.max_stack_depth);
    pool = pool_vec.data();
    stack = stack_vec.data();
#endif

    for (size_t i = 0; i < tree.num_nodes; ++i) {
        size_t idx = tree.post_order_indexes[i];
        if (tree.nodes[idx].shape_type == ShapeType::Sphere) {
            int start = pool_ptr;
            uint32_t my_count = 0;
#ifdef __CUDA_ARCH__
            if (pool_ptr + MAX_SPANS > MAX_POOL_SIZE) {
#else
            if (pool_ptr + MAX_SPANS > tree.max_pool_size) {
#endif
                count = 0;  // Overflow: handle by skipping
                return;
            }
            else {
                float data[MAX_SHAPE_DATA_SIZE];
                for (int j = 0; j < MAX_SHAPE_DATA_SIZE; ++j) data[j] = tree.data[idx * MAX_SHAPE_DATA_SIZE + j];
                Sphere s(data);
                s.material.color = Color(tree.red[idx], tree.green[idx], tree.blue[idx]);
                s.material.diffuse_coeff = tree.diffuse_coeff[idx];
                s.material.specular_coeff = tree.specular_coeff[idx];
                s.material.shininess = tree.shininess[idx];
                s.getSpans(ray, &pool[pool_ptr], my_count);
                pool_ptr += my_count;
            }
            // Push to stack
            stack[sp].start = start;
            stack[sp].count = my_count;
            ++sp;
            }
        else {
            // Pop right
            uint32_t right_count = stack[--sp].count;
            int right_start = stack[sp].start;
            // Pop left
            uint32_t left_count = stack[--sp].count;
            int left_start = stack[sp].start;
            // Compute result
            int result_start = pool_ptr;
            uint32_t result_count = 0;
#ifdef __CUDA_ARCH__
            if (pool_ptr + MAX_SPANS > MAX_POOL_SIZE) {
#else
            if (pool_ptr + MAX_SPANS > tree.max_pool_size) {
#endif
                count = 0;  // Overflow
                return;
            }
            if (tree.nodes[idx].op == CSGOp::UNION) {
                unionSpans(&pool[left_start], left_count, &pool[right_start], right_count, &pool[pool_ptr], result_count);
            }
            else if (tree.nodes[idx].op == CSGOp::INTERSECTION) {
                intersectionSpans(&pool[left_start], left_count, &pool[right_start], right_count, &pool[pool_ptr], result_count);
            }
            else if (tree.nodes[idx].op == CSGOp::DIFFERENCE) {
                differenceSpans(&pool[left_start], left_count, &pool[right_start], right_count, &pool[pool_ptr], result_count);
            }
            pool_ptr += result_count;
            // Push result
            stack[sp].start = result_start;
            stack[sp].count = result_count;
            ++sp;
            }
        }
    // Root spans
    count = stack[--sp].count;
    for (uint32_t i = 0; i < count; ++i) {
        spans[i] = pool[stack[sp].start + i];
    }
        }

__host__ __device__ Color trace(const Ray & ray, const Light & light, const FlatCSGTree & tree) {
    Span spans[MAX_SPANS];
    uint32_t count;
    getSpans(ray, spans, count, tree, 0);
    float min_t = 1e30f;
    Hit hit;
    for (uint32_t i = 0; i < count; ++i) {
        if (spans[i].t_entry > 0.001f && spans[i].t_entry < min_t) {
            min_t = spans[i].t_entry;
            hit = spans[i].entry_hit;
        }
    }
    if (min_t >= 1e30f) return Color(0, 0, 0);
    Vec3 point = ray.at(min_t);
    Vec3 L = -light.direction;
    float nl = (hit.normal.dot(L) > 0.f) ? hit.normal.dot(L) : 0.f;
    Color diffuse = hit.mat.color * hit.mat.diffuse_coeff * nl;
    Vec3 V = -ray.dir;
    Vec3 R = reflect(-L, hit.normal);
    float sp = (R.dot(V) > 0.f) ? R.dot(V) : 0.f;
    float spec = powf(sp, hit.mat.shininess) * hit.mat.specular_coeff;
    Color specular = Color(spec, spec, spec);
    Color ambient = hit.mat.color * 0.2f;
    return ambient + diffuse + specular;
}

__global__ void renderKernel(Color * image, const Camera cam, const Light light, const FlatCSGTree tree) {
    extern __shared__ char smem[];
    FlatCSGTree shared_tree;
    shared_tree.num_nodes = tree.num_nodes;
    char* ptr = smem;
    shared_tree.nodes = (FlatCSGNodeInfo*)ptr;
    ptr += shared_tree.num_nodes * sizeof(FlatCSGNodeInfo);
    shared_tree.data = (float*)ptr;
    ptr += shared_tree.num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float);
    shared_tree.red = (float*)ptr;
    ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.green = (float*)ptr;
    ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.blue = (float*)ptr;
    ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.diffuse_coeff = (float*)ptr;
    ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.specular_coeff = (float*)ptr;
    ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.shininess = (float*)ptr;
    ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.left_indexes = (size_t*)ptr;
    ptr += shared_tree.num_nodes * sizeof(size_t);
    shared_tree.right_indexes = (size_t*)ptr;
    ptr += shared_tree.num_nodes * sizeof(size_t);
    shared_tree.post_order_indexes = (size_t*)ptr;
    ptr += shared_tree.num_nodes * sizeof(size_t);

    // New: Copy dynamic sizes
    shared_tree.max_pool_size = tree.max_pool_size;
    shared_tree.max_stack_depth = tree.max_stack_depth;

    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int stride = blockDim.x * blockDim.y;

    // Copy nodes
    for (size_t i = tid; i < shared_tree.num_nodes; i += stride) {
        shared_tree.nodes[i] = tree.nodes[i];
    }

    // Copy data
    size_t data_size = shared_tree.num_nodes * MAX_SHAPE_DATA_SIZE;
    for (size_t i = tid; i < data_size; i += stride) {
        shared_tree.data[i] = tree.data[i];
    }

    // Copy red, green, blue, diffuse_coeff, specular_coeff, shininess, left_indexes, right_indexes, post_order_indexes
    for (size_t i = tid; i < shared_tree.num_nodes; i += stride) {
        shared_tree.red[i] = tree.red[i];
        shared_tree.green[i] = tree.green[i];
        shared_tree.blue[i] = tree.blue[i];
        shared_tree.diffuse_coeff[i] = tree.diffuse_coeff[i];
        shared_tree.specular_coeff[i] = tree.specular_coeff[i];
        shared_tree.shininess[i] = tree.shininess[i];
        shared_tree.left_indexes[i] = tree.left_indexes[i];
        shared_tree.right_indexes[i] = tree.right_indexes[i];
        shared_tree.post_order_indexes[i] = tree.post_order_indexes[i];
    }

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam.getWidth() || y >= cam.getHeight()) return;
    float s = (x + 0.5f) / cam.getWidth();
    float t = (y + 0.5f) / cam.getHeight();
    Ray ray = cam.getRay(s, t);
    image[y * cam.getWidth() + x] = trace(ray, light, shared_tree);
}

void copyTreeToDevice(const FlatCSGTree & h_tree, FlatCSGTree & d_tree) {
    cudaMalloc(&d_tree.nodes, h_tree.num_nodes * sizeof(FlatCSGNodeInfo));
    cudaMemcpy(d_tree.nodes, h_tree.nodes, h_tree.num_nodes * sizeof(FlatCSGNodeInfo), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.data, h_tree.num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float));
    cudaMemcpy(d_tree.data, h_tree.data, h_tree.num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.red, h_tree.num_nodes * sizeof(float));
    cudaMemcpy(d_tree.red, h_tree.red, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.green, h_tree.num_nodes * sizeof(float));
    cudaMemcpy(d_tree.green, h_tree.green, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.blue, h_tree.num_nodes * sizeof(float));
    cudaMemcpy(d_tree.blue, h_tree.blue, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.diffuse_coeff, h_tree.num_nodes * sizeof(float));
    cudaMemcpy(d_tree.diffuse_coeff, h_tree.diffuse_coeff, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.specular_coeff, h_tree.num_nodes * sizeof(float));
    cudaMemcpy(d_tree.specular_coeff, h_tree.specular_coeff, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.shininess, h_tree.num_nodes * sizeof(float));
    cudaMemcpy(d_tree.shininess, h_tree.shininess, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.left_indexes, h_tree.num_nodes * sizeof(size_t));
    cudaMemcpy(d_tree.left_indexes, h_tree.left_indexes, h_tree.num_nodes * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.right_indexes, h_tree.num_nodes * sizeof(size_t));
    cudaMemcpy(d_tree.right_indexes, h_tree.right_indexes, h_tree.num_nodes * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMalloc(&d_tree.post_order_indexes, h_tree.num_nodes * sizeof(size_t));
    cudaMemcpy(d_tree.post_order_indexes, h_tree.post_order_indexes, h_tree.num_nodes * sizeof(size_t), cudaMemcpyHostToDevice);
    d_tree.num_nodes = h_tree.num_nodes;

    d_tree.max_pool_size = h_tree.max_pool_size;
    d_tree.max_stack_depth = h_tree.max_stack_depth;
}

void freeDeviceTree(FlatCSGTree & d_tree) {
    cudaFree(d_tree.nodes);
    cudaFree(d_tree.data);
    cudaFree(d_tree.red);
    cudaFree(d_tree.green);
    cudaFree(d_tree.blue);
    cudaFree(d_tree.diffuse_coeff);
    cudaFree(d_tree.specular_coeff);
    cudaFree(d_tree.shininess);
    cudaFree(d_tree.left_indexes);
    cudaFree(d_tree.right_indexes);
    cudaFree(d_tree.post_order_indexes);
}

void freeHostTree(FlatCSGTree & tree) {
    delete[] tree.nodes;
    delete[] tree.data;
    delete[] tree.red;
    delete[] tree.green;
    delete[] tree.blue;
    delete[] tree.diffuse_coeff;
    delete[] tree.specular_coeff;
    delete[] tree.shininess;
    delete[] tree.left_indexes;
    delete[] tree.right_indexes;
    delete[] tree.post_order_indexes;
}

size_t computeMaxDepth(const FlatCSGTree & tree, size_t node_idx) {
    if (tree.nodes[node_idx].shape_type != ShapeType::TreeNode) return 1;
    size_t left = computeMaxDepth(tree, tree.left_indexes[node_idx]);
    size_t right = computeMaxDepth(tree, tree.right_indexes[node_idx]);
    return 1 + ((left > right) ? left : right);
}