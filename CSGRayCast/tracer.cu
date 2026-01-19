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

template <typename T>
__host__ __device__ void processLeafNode(
    const Ray& ray,
    const FlatCSGTree& tree,
    size_t idx,
    StridedSpan& pool,
    int& pool_ptr,
    StridedStack& stack,
    int& sp,
    uint32_t& count
) {
    // Bounds Check
    if (pool_ptr + MAX_SPANS > tree.max_pool_size) { count = 0; return; }

    // Load Data from Flat Array
    float data[MAX_SHAPE_DATA_SIZE];
    // Unrolling this loop often helps performance
#pragma unroll
    for (int j = 0; j < MAX_SHAPE_DATA_SIZE; ++j) {
        data[j] = tree.data[idx * MAX_SHAPE_DATA_SIZE + j];
    }

    // Construct the Shape (T)
    T shape(data);

    // Set Material ID (optimized: don't copy full material struct)
    shape.material_id = (int)idx;

    // Compute Intersections
    int start = pool_ptr;
    uint32_t my_count = 0;

    // Pass the specific pointer within the strided pool
    shape.getSpans(ray, pool.at(pool_ptr), my_count);
    pool_ptr += my_count;

    // Push to Stack
    if (sp >= tree.max_stack_depth) { count = 0; return; }
    stack[sp].start = start;
    stack[sp].count = my_count;
    ++sp;
}

__host__ __device__ Vec3 reflect(const Vec3& I, const Vec3& N) {
    return I - N * 2.f * I.dot(N);
}

// Helper: Copies strided data to local contiguous array
__host__ __device__ void copyToLocal(const StridedSpan& src, uint32_t count, Span* dst) {
    for (uint32_t i = 0; i < count; ++i) {
        dst[i] = src[i];
    }
}

__host__ __device__ void unionSpans(
    const StridedSpan& left, uint32_t left_count,
    const StridedSpan& right, uint32_t right_count,
    StridedSpan& result, uint32_t& result_count)
{
    result_count = 0;
    if (left_count == 0 && right_count == 0) return;

    uint32_t li = 0;
    uint32_t ri = 0;

    // Registers for the "current active union interval"
    Span current;

    // 1. Initialize 'current' with the earliest span from either left or right
    bool take_left = false;
    if (li < left_count && ri < right_count) {
        if (left[li].t_entry < right[ri].t_entry) take_left = true;
    }
    else if (li < left_count) {
        take_left = true;
    }
    // else take_left = false (implies taking right)

    if (take_left) {
        current = left[li++];
    }
    else {
        current = right[ri++];
    }

    // 2. Stream through the remaining spans
    while (li < left_count || ri < right_count) {
        Span next;

        // Determine which span comes next in time
        bool next_is_left = false;
        if (li < left_count && ri < right_count) {
            if (left[li].t_entry < right[ri].t_entry) next_is_left = true;
        }
        else if (li < left_count) {
            next_is_left = true;
        }

        // Load the candidate
        if (next_is_left) {
            next = left[li++];
        }
        else {
            next = right[ri++];
        }

        // 3. Merge Logic
        // If the next span starts before (or exactly when) the current one ends: Overlap
        if (next.t_entry <= current.t_exit + 1e-6f) {
            // Extend the current interval if the new one goes further
            if (next.t_exit > current.t_exit) {
                current.t_exit = next.t_exit;
                current.exit_hit = next.exit_hit;
            }
        }
        else {
            // Disjoint: Write the finished 'current' to result
            result[result_count++] = current;

            // Start a new interval
            current = next;
        }
    }

    // 4. Write the final pending interval
    result[result_count++] = current;
}

__host__ __device__ void intersectionSpans(
    const StridedSpan& left, uint32_t left_count,
    const StridedSpan& right, uint32_t right_count,
    StridedSpan& result, uint32_t& result_count)
{
    result_count = 0;
    uint32_t li = 0;
    uint32_t ri = 0;

    while (li < left_count && ri < right_count) {
        // Load data into registers
        Span l_span = left[li];
        Span r_span = right[ri];

        // Determine overlap interval
        float start = (l_span.t_entry > r_span.t_entry) ? l_span.t_entry : r_span.t_entry;
        float end = (l_span.t_exit < r_span.t_exit) ? l_span.t_exit : r_span.t_exit;

        // If there is a valid overlap
        if (start < end) {
            // Logic: The Entry hit is determined by whoever started later (max).
            //        The Exit hit is determined by whoever ended earlier (min).
            Hit e_hit = (l_span.t_entry > r_span.t_entry) ? l_span.entry_hit : r_span.entry_hit;
            Hit x_hit = (l_span.t_exit < r_span.t_exit) ? l_span.exit_hit : r_span.exit_hit;

            // Direct write to global result
            result[result_count].t_entry = start;
            result[result_count].entry_hit = e_hit;
            result[result_count].t_exit = end;
            result[result_count].exit_hit = x_hit;
            ++result_count;
        }

        // Advance the pointer of the span that ends first.
        // It cannot possibly overlap with any future span from the other list.
        if (l_span.t_exit < r_span.t_exit) {
            li++;
        }
        else {
            ri++;
        }
    }
}

__host__ __device__ void differenceSpans(
    const StridedSpan& left, uint32_t left_count,
    const StridedSpan& right, uint32_t right_count,
    StridedSpan& result, uint32_t& result_count)
{
    result_count = 0;
    uint32_t ri_base = 0; // Tracks the first potential 'Right' span for the current 'Left'

    for (uint32_t li = 0; li < left_count; ++li) {
        Span curr = left[li]; // Load current Left span into register

        // Iterate through Right spans that might overlap 'curr'
        for (uint32_t ri = ri_base; ri < right_count; ++ri) {
            Span sub = right[ri];

            // optimization: If Right span is completely behind Current Left, 
            // we never need to check it again for future Left spans.
            if (sub.t_exit <= curr.t_entry) {
                ri_base = ri + 1;
                continue;
            }

            // optimization: If Right span is completely ahead, it doesn't affect Current Left.
            // But we must NOT increment ri_base, as it might affect the NEXT Left span.
            if (sub.t_entry >= curr.t_exit) {
                break;
            }

            // --- Overlap Logic ---

            // 1. If there is a gap between Current Start and Subtractor Start, keep that segment.
            if (sub.t_entry > curr.t_entry) {
                Span valid_segment;
                valid_segment.t_entry = curr.t_entry;
                valid_segment.entry_hit = curr.entry_hit;

                valid_segment.t_exit = sub.t_entry;
                valid_segment.exit_hit = sub.entry_hit;
                valid_segment.exit_hit.normal = -valid_segment.exit_hit.normal; // Invert normal

                result[result_count++] = valid_segment;
            }

            // 2. Cut the beginning of 'curr' by moving t_entry to the end of the subtractor
            if (sub.t_exit > curr.t_entry) {
                curr.t_entry = sub.t_exit;
                curr.entry_hit = sub.exit_hit;
                curr.entry_hit.normal = -curr.entry_hit.normal; // Invert normal
            }

            // 3. If 'curr' has been completely eaten, stop processing it
            if (curr.t_entry >= curr.t_exit) {
                break;
            }
        }

        // If anything remains of the Left span after checking all Right spans, save it.
        if (curr.t_entry < curr.t_exit) {
            result[result_count++] = curr;
        }
    }
}

__host__ __device__ void getSpans(const Ray& ray, Span* spans, uint32_t& count, const FlatCSGTree& tree, size_t node_idx, StridedSpan thread_pool, StridedStack thread_stack) {
    count = 0;
    if (tree.num_nodes == 0) return;

    StridedSpan pool = thread_pool;
    StridedStack stack = thread_stack;

    int pool_ptr = 0;
    int sp = 0;

#ifndef __CUDA_ARCH__
    // Fallback for CPU execution (contiguous memory)
    std::vector<Span> pool_vec;
    std::vector<StackEntry> stack_vec;
    if (!pool.is_valid() || !stack.is_valid()) {
        pool_vec.resize(tree.max_pool_size);
        stack_vec.resize(tree.max_stack_depth);
        pool = StridedSpan(pool_vec.data(), 1);
        stack = StridedStack(stack_vec.data(), 1);
    }
#endif

    for (size_t i = 0; i < tree.num_nodes; ++i) {
        size_t idx = tree.post_order_indexes[i];
        ShapeType type = tree.nodes[idx].shape_type;

        // Dispatch based on type, but use the templated helper
        if (type == ShapeType::Sphere) {
            processLeafNode<Sphere>(ray, tree, idx, pool, pool_ptr, stack, sp, count);
        }
        else if (type == ShapeType::Cuboid) {
            processLeafNode<Cuboid>(ray, tree, idx, pool, pool_ptr, stack, sp, count);
        }
        else if (type == ShapeType::Cylinder) {
            processLeafNode<Cylinder>(ray, tree, idx, pool, pool_ptr, stack, sp, count);
        }
        else if (type == ShapeType::Cone) {
            processLeafNode<Cone>(ray, tree, idx, pool, pool_ptr, stack, sp, count);
        }
        else {
            // TreeNode (Operators: UNION, INTERSECTION, DIFFERENCE)
            if (sp < 2) { count = 0; return; }
            uint32_t right_count = stack[--sp].count;
            int right_start = stack[sp].start;
            uint32_t left_count = stack[--sp].count;
            int left_start = stack[sp].start;

            int result_start = pool_ptr;
            uint32_t result_count = 0;

            if (pool_ptr + MAX_SPANS > tree.max_pool_size) { count = 0; return; }

            StridedSpan left_span(pool.at(left_start), pool.stride);
            StridedSpan right_span(pool.at(right_start), pool.stride);
            StridedSpan result_span(pool.at(pool_ptr), pool.stride);

            if (tree.nodes[idx].op == CSGOp::UNION)
                unionSpans(left_span, left_count, right_span, right_count, result_span, result_count);
            else if (tree.nodes[idx].op == CSGOp::INTERSECTION)
                intersectionSpans(left_span, left_count, right_span, right_count, result_span, result_count);
            else if (tree.nodes[idx].op == CSGOp::DIFFERENCE)
                differenceSpans(left_span, left_count, right_span, right_count, result_span, result_count);

            pool_ptr += result_count;
            stack[sp].start = result_start;
            stack[sp].count = result_count;
            ++sp;
        }

        // Safety check if the helper set count to 0 (error case)
        if (count == 0 && sp == 0 && pool_ptr == 0) return;
    }
    if (sp > 0) {
        count = stack[--sp].count;
        // Copy from strided pool to the final linear result buffer
        for (uint32_t i = 0; i < count; ++i) spans[i] = pool[stack[sp].start + i];
    }
}

__host__ __device__ Color trace(const Ray& ray, const Light& light, const FlatCSGTree& tree, StridedSpan thread_pool, StridedStack thread_stack) {
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

    // Look up material data at the very end
    int mat_id = hit.node_id;

    Color mat_color(0.5f, 0.5f, 0.5f);
    float diff_coeff = 0.5f;
    float spec_coeff = 0.5f;
    float shininess = 10.0f;

    if (mat_id >= 0) {
        mat_color = Color(tree.red[mat_id], tree.green[mat_id], tree.blue[mat_id]);
        diff_coeff = tree.diffuse_coeff[mat_id];
        spec_coeff = tree.specular_coeff[mat_id];
        shininess = tree.shininess[mat_id];
    }

    Vec3 point = ray.at(min_t);
    Vec3 L = -light.direction;
    float nl = (hit.normal.dot(L) > 0.f) ? hit.normal.dot(L) : 0.f;
    Color diffuse = mat_color * diff_coeff * nl;
    Vec3 V = -ray.dir;
    Vec3 R = reflect(-L, hit.normal);
    float sp = (R.dot(V) > 0.f) ? R.dot(V) : 0.f;
    float spec = powf(sp, shininess) * spec_coeff;
    return (mat_color * 0.2f) + diffuse + Color(spec, spec, spec);
}

__global__ void renderKernel(Color* image, const Camera cam, const Light light, const FlatCSGTree tree,
    Span* global_pool, StackEntry* global_stack,
    size_t pixel_offset, size_t batch_size, size_t total_pixels)
{
    // Use Shared Memory for Tree Data
    extern __shared__ char smem[];
    FlatCSGTree shared_tree;
    shared_tree.num_nodes = tree.num_nodes;
    char* ptr = smem;

    // Allocation alignment
    shared_tree.left_indexes = (size_t*)ptr; ptr += shared_tree.num_nodes * sizeof(size_t);
    shared_tree.right_indexes = (size_t*)ptr; ptr += shared_tree.num_nodes * sizeof(size_t);
    shared_tree.post_order_indexes = (size_t*)ptr; ptr += shared_tree.num_nodes * sizeof(size_t);
    shared_tree.nodes = (FlatCSGNodeInfo*)ptr; ptr += shared_tree.num_nodes * sizeof(FlatCSGNodeInfo);
    shared_tree.data = (float*)ptr; ptr += shared_tree.num_nodes * MAX_SHAPE_DATA_SIZE * sizeof(float);
    shared_tree.red = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.green = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.blue = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.diffuse_coeff = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.specular_coeff = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);
    shared_tree.shininess = (float*)ptr; ptr += shared_tree.num_nodes * sizeof(float);

    shared_tree.max_pool_size = tree.max_pool_size;
    shared_tree.max_stack_depth = tree.max_stack_depth;

    unsigned int local_id = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int stride = blockDim.x * blockDim.y;

    for (size_t i = local_id; i < shared_tree.num_nodes; i += stride) {
        shared_tree.nodes[i] = tree.nodes[i];
        shared_tree.left_indexes[i] = tree.left_indexes[i];
        shared_tree.right_indexes[i] = tree.right_indexes[i];
        shared_tree.post_order_indexes[i] = tree.post_order_indexes[i];
        shared_tree.red[i] = tree.red[i];
        shared_tree.green[i] = tree.green[i];
        shared_tree.blue[i] = tree.blue[i];
        shared_tree.diffuse_coeff[i] = tree.diffuse_coeff[i];
        shared_tree.specular_coeff[i] = tree.specular_coeff[i];
        shared_tree.shininess[i] = tree.shininess[i];
    }

    size_t data_size = shared_tree.num_nodes * MAX_SHAPE_DATA_SIZE;
    for (size_t i = local_id; i < data_size; i += stride) {
        shared_tree.data[i] = tree.data[i];
    }
    __syncthreads();

    // --- BATCH PROCESSING LOGIC ---
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    size_t global_idx = pixel_offset + tid;
    if (global_idx >= total_pixels) return;

    int width = cam.getWidth();
    int x = global_idx % width;
    int y = global_idx / width;

    // --- STRIDED MEMORY SETUP ---
    // Start at: global_base + tid
    // Stride: batch_size
    // This creates an interleaved memory pattern:
    // Thread 0: [0, batch, 2*batch ...]
    // Thread 1: [1, batch+1, 2*batch+1 ...]
    StridedSpan my_pool(global_pool + tid, batch_size);
    StridedStack my_stack(global_stack + tid, batch_size);

    float s = (x + 0.5f) / width;
    float t = (y + 0.5f) / cam.getHeight();
    Ray ray = cam.getRay(s, t);

    image[global_idx] = trace(ray, light, shared_tree, my_pool, my_stack);
}

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