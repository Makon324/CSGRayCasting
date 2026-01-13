#include "shape.h"
#include "tracer.h"
#include "csg.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <iostream>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << msg << std::endl;
        exit(1);
    }
}

__host__ __device__ Vec3 reflect(const Vec3& I, const Vec3& N) {
    return I - N * 2.f * I.dot(N);
}

// [unionSpans, intersectionSpans, differenceSpans implementation remains identical to previous version]
// ... (Including them here for completeness if you copy-paste, but omitting their body to save space if you prefer. 
//      Below I assume the standard implementation you had is used.)

__host__ __device__ void unionSpans(const Span* left, uint32_t left_count, const Span* right, uint32_t right_count, Span* result, uint32_t& result_count) {
    result_count = 0;
    if (left_count == 0 && right_count == 0) return;
    Span all[2 * MAX_SPANS];
    uint32_t all_count = 0;
    uint32_t li = 0, ri = 0;
    while (li < left_count && ri < right_count) {
        if (left[li].t_entry < right[ri].t_entry) all[all_count++] = left[li++];
        else all[all_count++] = right[ri++];
    }
    while (li < left_count) all[all_count++] = left[li++];
    while (ri < right_count) all[all_count++] = right[ri++];
    float current_start = all[0].t_entry;
    float current_end = all[0].t_exit;
    Hit entry_hit = all[0].entry_hit;
    Hit exit_hit = all[0].exit_hit;
    for (uint32_t i = 1; i < all_count; ++i) {
        if (all[i].t_entry <= current_end + 1e-6f) {
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
        if (current.t_entry < current.t_exit) temp[temp_count++] = current;
    }
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

__host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count, const FlatCSGTree& tree, size_t node_idx, Span* thread_pool, StackEntry* thread_stack) {
    count = 0;
    if (tree.num_nodes == 0) return;

    Span* pool = thread_pool;
    StackEntry* stack = thread_stack;

    int pool_ptr = 0;
    int sp = 0;

#ifndef __CUDA_ARCH__
    std::vector<Span> pool_vec;
    std::vector<StackEntry> stack_vec;
    if (!pool || !stack) {
        pool_vec.resize(tree.max_pool_size);
        stack_vec.resize(tree.max_stack_depth);
        pool = pool_vec.data();
        stack = stack_vec.data();
    }
#endif

    for (size_t i = 0; i < tree.num_nodes; ++i) {
        size_t idx = tree.post_order_indexes[i];

        if (tree.nodes[idx].shape_type == ShapeType::Sphere) {
            int start = pool_ptr;
            uint32_t my_count = 0;

            if (pool_ptr + MAX_SPANS > tree.max_pool_size) { count = 0; return; }

            float data[MAX_SHAPE_DATA_SIZE];
            for (int j = 0; j < MAX_SHAPE_DATA_SIZE; ++j) data[j] = tree.data[idx * MAX_SHAPE_DATA_SIZE + j];
            Sphere s(data);
            s.material.color = Color(tree.red[idx], tree.green[idx], tree.blue[idx]);
            s.material.diffuse_coeff = tree.diffuse_coeff[idx];
            s.material.specular_coeff = tree.specular_coeff[idx];
            s.material.shininess = tree.shininess[idx];
            s.getSpans(ray, &pool[pool_ptr], my_count);
            pool_ptr += my_count;

            if (sp >= tree.max_stack_depth) { count = 0; return; }
            stack[sp].start = start;
            stack[sp].count = my_count;
            ++sp;
        }
        else {
            if (sp < 2) { count = 0; return; }
            uint32_t right_count = stack[--sp].count;
            int right_start = stack[sp].start;
            uint32_t left_count = stack[--sp].count;
            int left_start = stack[sp].start;

            int result_start = pool_ptr;
            uint32_t result_count = 0;

            if (pool_ptr + MAX_SPANS > tree.max_pool_size) { count = 0; return; }

            if (tree.nodes[idx].op == CSGOp::UNION) unionSpans(&pool[left_start], left_count, &pool[right_start], right_count, &pool[pool_ptr], result_count);
            else if (tree.nodes[idx].op == CSGOp::INTERSECTION) intersectionSpans(&pool[left_start], left_count, &pool[right_start], right_count, &pool[pool_ptr], result_count);
            else if (tree.nodes[idx].op == CSGOp::DIFFERENCE) differenceSpans(&pool[left_start], left_count, &pool[right_start], right_count, &pool[pool_ptr], result_count);

            pool_ptr += result_count;
            stack[sp].start = result_start;
            stack[sp].count = result_count;
            ++sp;
        }
    }
    if (sp > 0) {
        count = stack[--sp].count;
        for (uint32_t i = 0; i < count; ++i) spans[i] = pool[stack[sp].start + i];
    }
}

__host__ __device__ Color trace(const Ray& ray, const Light& light, const FlatCSGTree& tree, Span* thread_pool, StackEntry* thread_stack) {
    Span spans[MAX_SPANS];
    uint32_t count;
    getSpans(ray, spans, count, tree, 0, thread_pool, thread_stack);

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
    return (hit.mat.color * 0.2f) + diffuse + Color(spec, spec, spec);
}

// UPDATED RENDER KERNEL
__global__ void renderKernel(Color* image, const Camera cam, const Light light, const FlatCSGTree tree,
    Span* global_pool, StackEntry* global_stack,
    size_t pixel_offset, size_t batch_size, size_t total_pixels)
{
    // Use Shared Memory for Tree Data (Same as before)
    extern __shared__ char smem[];
    FlatCSGTree shared_tree;
    // ... [Shared memory pointer setup remains same] ...
    shared_tree.num_nodes = tree.num_nodes;
    char* ptr = smem;
    shared_tree.nodes = (FlatCSGNodeInfo*)ptr; ptr += shared_tree.num_nodes * sizeof(FlatCSGNodeInfo);
    shared_tree.data = (float*)ptr; ptr += shared_tree.num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float);
    shared_tree.red = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.green = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.blue = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.diffuse_coeff = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.specular_coeff = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.shininess = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.left_indexes = (size_t*)ptr; ptr += shared_tree.num_nodes * sizeof(size_t);
    shared_tree.right_indexes = (size_t*)ptr; ptr += shared_tree.num_nodes * sizeof(size_t);
    shared_tree.post_order_indexes = (size_t*)ptr; ptr += shared_tree.num_nodes * sizeof(size_t);

    shared_tree.max_pool_size = tree.max_pool_size;
    shared_tree.max_stack_depth = tree.max_stack_depth;

    unsigned int local_id = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int stride = blockDim.x * blockDim.y;
    for (size_t i = local_id; i < shared_tree.num_nodes; i += stride) shared_tree.nodes[i] = tree.nodes[i];
    size_t data_size = shared_tree.num_nodes * MAX_SHAPE_DATA_SIZE;
    for (size_t i = local_id; i < data_size; i += stride) shared_tree.data[i] = tree.data[i];
    for (size_t i = local_id; i < shared_tree.num_nodes; i += stride) {
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

    // --- BATCH PROCESSING LOGIC ---

    // 1. Calculate the Linear Thread ID for this specific batch launch
    //    We interpret the grid as 1D linear for simplicity in batching
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Boundary Check: Ensure we don't go past the batch size
    if (tid >= batch_size) return;

    // 3. Calculate the actual global pixel index
    size_t global_idx = pixel_offset + tid;

    // 4. Boundary Check: Ensure we don't go past the total image size
    if (global_idx >= total_pixels) return;

    // 5. Map linear global index back to (X, Y) for Ray Generation
    int width = cam.getWidth();
    int x = global_idx % width;
    int y = global_idx / width;

    // 6. Memory Slicing: Use the LOCAL 'tid' (0 to batch_size) to access the scratch buffer.
    //    The scratch buffer is reused for every batch, so we always index it from 0.
    Span* my_pool = global_pool + (static_cast<size_t>(tid) * shared_tree.max_pool_size);
    StackEntry* my_stack = global_stack + (static_cast<size_t>(tid) * shared_tree.max_stack_depth);

    float s = (x + 0.5f) / width;
    float t = (y + 0.5f) / cam.getHeight();
    Ray ray = cam.getRay(s, t);

    // 7. Write to global image memory using the true global index
    image[global_idx] = trace(ray, light, shared_tree, my_pool, my_stack);
}

// [Copy/Free helper functions remain identical]
void copyTreeToDevice(const FlatCSGTree& h_tree, FlatCSGTree& d_tree) {
    checkCudaError(cudaMalloc(&d_tree.nodes, h_tree.num_nodes * sizeof(FlatCSGNodeInfo)), "cudaMalloc d_tree.nodes");
    checkCudaError(cudaMemcpy(d_tree.nodes, h_tree.nodes, h_tree.num_nodes * sizeof(FlatCSGNodeInfo), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.nodes");
    checkCudaError(cudaMalloc(&d_tree.data, h_tree.num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float)), "cudaMalloc d_tree.data");
    checkCudaError(cudaMemcpy(d_tree.data, h_tree.data, h_tree.num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.data");
    checkCudaError(cudaMalloc(&d_tree.red, h_tree.num_nodes * sizeof(float)), "cudaMalloc d_tree.red");
    checkCudaError(cudaMemcpy(d_tree.red, h_tree.red, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.red");
    checkCudaError(cudaMalloc(&d_tree.green, h_tree.num_nodes * sizeof(float)), "cudaMalloc d_tree.green");
    checkCudaError(cudaMemcpy(d_tree.green, h_tree.green, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.green");
    checkCudaError(cudaMalloc(&d_tree.blue, h_tree.num_nodes * sizeof(float)), "cudaMalloc d_tree.blue");
    checkCudaError(cudaMemcpy(d_tree.blue, h_tree.blue, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.blue");
    checkCudaError(cudaMalloc(&d_tree.diffuse_coeff, h_tree.num_nodes * sizeof(float)), "cudaMalloc d_tree.diffuse_coeff");
    checkCudaError(cudaMemcpy(d_tree.diffuse_coeff, h_tree.diffuse_coeff, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.diffuse_coeff");
    checkCudaError(cudaMalloc(&d_tree.specular_coeff, h_tree.num_nodes * sizeof(float)), "cudaMalloc d_tree.specular_coeff");
    checkCudaError(cudaMemcpy(d_tree.specular_coeff, h_tree.specular_coeff, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.specular_coeff");
    checkCudaError(cudaMalloc(&d_tree.shininess, h_tree.num_nodes * sizeof(float)), "cudaMalloc d_tree.shininess");
    checkCudaError(cudaMemcpy(d_tree.shininess, h_tree.shininess, h_tree.num_nodes * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.shininess");
    checkCudaError(cudaMalloc(&d_tree.left_indexes, h_tree.num_nodes * sizeof(size_t)), "cudaMalloc d_tree.left_indexes");
    checkCudaError(cudaMemcpy(d_tree.left_indexes, h_tree.left_indexes, h_tree.num_nodes * sizeof(size_t), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.left_indexes");
    checkCudaError(cudaMalloc(&d_tree.right_indexes, h_tree.num_nodes * sizeof(size_t)), "cudaMalloc d_tree.right_indexes");
    checkCudaError(cudaMemcpy(d_tree.right_indexes, h_tree.right_indexes, h_tree.num_nodes * sizeof(size_t), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.right_indexes");
    checkCudaError(cudaMalloc(&d_tree.post_order_indexes, h_tree.num_nodes * sizeof(size_t)), "cudaMalloc d_tree.post_order_indexes");
    checkCudaError(cudaMemcpy(d_tree.post_order_indexes, h_tree.post_order_indexes, h_tree.num_nodes * sizeof(size_t), cudaMemcpyHostToDevice), "cudaMemcpy d_tree.post_order_indexes");
    d_tree.num_nodes = h_tree.num_nodes;
    d_tree.max_pool_size = h_tree.max_pool_size;
    d_tree.max_stack_depth = h_tree.max_stack_depth;
}

void freeDeviceTree(FlatCSGTree& d_tree) {
    checkCudaError(cudaFree(d_tree.nodes), "cudaFree d_tree.nodes");
    checkCudaError(cudaFree(d_tree.data), "cudaFree d_tree.data");
    checkCudaError(cudaFree(d_tree.red), "cudaFree d_tree.red");
    checkCudaError(cudaFree(d_tree.green), "cudaFree d_tree.green");
    checkCudaError(cudaFree(d_tree.blue), "cudaFree d_tree.blue");
    checkCudaError(cudaFree(d_tree.diffuse_coeff), "cudaFree d_tree.diffuse_coeff");
    checkCudaError(cudaFree(d_tree.specular_coeff), "cudaFree d_tree.specular_coeff");
    checkCudaError(cudaFree(d_tree.shininess), "cudaFree d_tree.shininess");
    checkCudaError(cudaFree(d_tree.left_indexes), "cudaFree d_tree.left_indexes");
    checkCudaError(cudaFree(d_tree.right_indexes), "cudaFree d_tree.right_indexes");
    checkCudaError(cudaFree(d_tree.post_order_indexes), "cudaFree d_tree.post_order_indexes");
}

void freeHostTree(FlatCSGTree& tree) {
    delete[] tree.nodes; delete[] tree.data; delete[] tree.red; delete[] tree.green; delete[] tree.blue;
    delete[] tree.diffuse_coeff; delete[] tree.specular_coeff; delete[] tree.shininess;
    delete[] tree.left_indexes; delete[] tree.right_indexes; delete[] tree.post_order_indexes;
}
size_t computeMaxDepth(const FlatCSGTree& tree, size_t node_idx) {
    if (tree.nodes[node_idx].shape_type != ShapeType::TreeNode) return 1;
    size_t left = computeMaxDepth(tree, tree.left_indexes[node_idx]);
    size_t right = computeMaxDepth(tree, tree.right_indexes[node_idx]);
    return 1 + ((left > right) ? left : right);
}