//float4 对齐版本
#include <stdint.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __inline__ uint64_t make_key(float prob, int64_t col) {
    return (((uint64_t)__float_as_uint(prob)) << 32) |
           (uint64_t)(0xffffffffu - (uint32_t)col);
}

__device__ __inline__ bool accept_key(float p, int64_t col,
                                      uint64_t mask,
                                      uint64_t current_mask,
                                      uint64_t now_flag) {
    uint64_t key = make_key(p, col);
    return ((key & mask) == current_mask) && ((key & now_flag) != 0);
}

__device__ __inline__ bool keep_key(float p, int64_t col, uint64_t threshold) {
    return make_key(p, col) >= threshold;
}

__device__ __inline__ float block_reduce_sum(float val) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    val = cg::reduce(tile, val, cg::plus<float>());

    __shared__ float warp_sums[32];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        val = cg::reduce(tile, val, cg::plus<float>());
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    return warp_sums[0];
}

__global__ void topp_float4(const float* __restrict__ probs,
                            const float* __restrict__ top_p,
                            float* __restrict__ renorm_probs,
                            int64_t batch_size,
                            int64_t num_classes) {
    int tid = threadIdx.x;

    __shared__ uint64_t mask_share;
    __shared__ uint64_t current_mask_share;
    __shared__ float   accept_sum;

    if (tid == 0) {
        mask_share = 0ull;
        current_mask_share = 0ull;
        accept_sum = 0.0f;
    }
    __syncthreads();

    const int64_t row_offset = (int64_t)blockIdx.x * num_classes;
    const float row_top_p = top_p[blockIdx.x];
    const float* row_probs = probs + row_offset;

    // 只有行首 16B 对齐时才启用 float4
    const bool aligned = ((reinterpret_cast<uintptr_t>(row_probs) & 15) == 0);
    const int64_t vec_end = aligned ? (num_classes & ~3LL) : 0;
    const int64_t stride4 = (int64_t)blockDim.x * 4;

    // ---------- 1. 从高位到低位寻找分界点 ----------
    for (int bit = 63; bit >= 0; --bit) {
        const uint64_t current_mask = current_mask_share;
        const uint64_t mask = mask_share;
        const uint64_t now_flag = 1ull << bit;

        float local_sum = 0.0f;

        // float4 向量化部分 
        for (int64_t col4 = (int64_t)tid * 4; col4 < vec_end; col4 += stride4) {
            float4 v = *reinterpret_cast<const float4*>(row_probs + col4);
            if (accept_key(v.x, col4 + 0, mask, current_mask, now_flag)) local_sum += v.x;
            if (accept_key(v.y, col4 + 1, mask, current_mask, now_flag)) local_sum += v.y;
            if (accept_key(v.z, col4 + 2, mask, current_mask, now_flag)) local_sum += v.z;
            if (accept_key(v.w, col4 + 3, mask, current_mask, now_flag)) local_sum += v.w;
        }

        // 尾部标量处理
        for (int64_t col = vec_end + tid; col < num_classes; col += blockDim.x) {
            float p = row_probs[col];
            if (accept_key(p, col, mask, current_mask, now_flag)) {
                local_sum += p;
            }
        }

        const float total_sum = block_reduce_sum(local_sum);

        if (tid == 0) {
            if (total_sum > 0.0f && accept_sum + total_sum <= row_top_p) {
                accept_sum += total_sum;
            } else if (total_sum > 0.0f) {
                current_mask_share |= now_flag;
            }
            mask_share |= now_flag;
        }
        __syncthreads();
    }

    // ---------- 2. 重归一化 ----------
    const uint64_t threshold = current_mask_share;

    float final_sum = 0.0f;

    // 向量化求 final_sum
    for (int64_t col4 = (int64_t)tid * 4; col4 < vec_end; col4 += stride4) {
        float4 v = *reinterpret_cast<const float4*>(row_probs + col4);
        if (keep_key(v.x, col4 + 0, threshold)) final_sum += v.x;
        if (keep_key(v.y, col4 + 1, threshold)) final_sum += v.y;
        if (keep_key(v.z, col4 + 2, threshold)) final_sum += v.z;
        if (keep_key(v.w, col4 + 3, threshold)) final_sum += v.w;
    }

    for (int64_t col = vec_end + tid; col < num_classes; col += blockDim.x) {
        float p = row_probs[col];
        if (keep_key(p, col, threshold)) final_sum += p;
    }

    final_sum = block_reduce_sum(final_sum);
    const float inv = (final_sum > 0.0f) ? (1.0f / final_sum) : 0.0f;

    // 写回用标量，避免处理写对齐
    for (int64_t col = tid; col < num_classes; col += blockDim.x) {
        float p = row_probs[col];
        renorm_probs[row_offset + col] =
            keep_key(p, col, threshold) ? (p * inv) : 0.0f;
    }
}

extern "C" void run_kernel(const float* probs,
                                  const float* top_p,
                                  float* renorm_probs,
                                  int64_t batch_size,
                                  int64_t num_classes) {
    topp_float4<<<batch_size, 1024>>>(probs, top_p, renorm_probs,
                                      batch_size, num_classes);
}