#include <stdint.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __inline__ uint64_t make_key(float prob, int64_t col) {
    return (((uint64_t)__float_as_uint(prob)) << 32) |
           (uint64_t)(0xffffffffu - (uint32_t)col);
}

__device__ __inline__ int block_reduce_sum(int val) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    // 第一步：warp 内归约
    val = cg::reduce(tile, val, cg::plus<int>());

    __shared__ int warp_sums[32];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();

    // 第二步：第一个 warp 做二次归约
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? warp_sums[lane_id] : 0;
        val = cg::reduce(tile, val, cg::plus<int>());
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    return warp_sums[0];   // 广播给所有线程
}
__device__ __inline__ float block_reduce_sum_float(float val) {
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

// ============================== 主 kernel ==============================
__global__ void topk(const float* __restrict__ probs,
                     const int32_t* __restrict__ top_k,
                     float* __restrict__ renorm_probs,
                     int64_t batch_size,
                     int64_t num_classes) {
    int tid = threadIdx.x;

    __shared__ uint64_t mask_share;
    __shared__ uint64_t current_mask_share;
    __shared__ int32_t   accept_cnt;

    if (tid == 0) {
        mask_share = 0ull;
        current_mask_share = 0ull;
        accept_cnt = 0;
    }
    __syncthreads();

    const int64_t row_offset = (int64_t)blockIdx.x * num_classes;
    const int row_top_k = top_k[blockIdx.x];

    // ---------- 1. 从高位到低位寻找分界点 ----------
    for (int bit = 63; bit >= 0; --bit) {
        const uint64_t current_mask = current_mask_share;
        const uint64_t mask = mask_share;
        const uint64_t now_flag = 1ull << bit;

        int local_cnt = 0;
        for (int64_t col = tid; col < num_classes; col += blockDim.x) {
            float p = probs[row_offset + col];
            uint64_t key = make_key(p, col);
            if (((key & mask) == current_mask) && ((key & now_flag) != 0)) {
                local_cnt += 1;
            }
        }
        const int total_cnt = block_reduce_sum(local_cnt);

        if (tid == 0) {
            if (accept_cnt + total_cnt < row_top_k) {
                accept_cnt += total_cnt;              // 包含该位
            } else {
                current_mask_share |= now_flag;       // 分界点在此位
            }
            mask_share |= now_flag;
        }
        __syncthreads();
    }

    // ---------- 2. 重归一化 ----------
    const uint64_t threshold = current_mask_share;

    float final_sum = 0.0f;
    for (int64_t col = tid; col < num_classes; col += blockDim.x) {
        float p = probs[row_offset + col];
        uint64_t key = make_key(p, col);
        if (key >= threshold) final_sum += p;
    }
    final_sum = block_reduce_sum_float(final_sum);

    const float inv = (final_sum > 0.0f) ? (1.0f / final_sum) : 0.0f;
    for (int64_t col = tid; col < num_classes; col += blockDim.x) {
        float p = probs[row_offset + col];
        uint64_t key = make_key(p, col);
        renorm_probs[row_offset + col] = (key >= threshold) ? (p * inv) : 0.0f;
    }
}

extern "C" void run_kernel(const float* probs,
                           const int32_t* top_k,
                           float* renorm_probs,
                           int64_t batch_size,
                           int64_t num_classes) {
    topk<<<batch_size, 1024>>>(probs, top_k, renorm_probs, batch_size, num_classes);
}