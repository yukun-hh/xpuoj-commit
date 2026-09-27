#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

constexpr int BM = 32;        // block tile M
constexpr int BN = 32;        // block tile N
constexpr int BK = 128;        // block tile K
constexpr int PAD = 4;        // shared memory padding，缓解 bank 冲突

constexpr int WARPS_M = BM / 16;   // 4
constexpr int WARPS_N = BN / 16;   // 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N;   // 16
constexpr int THREADS   = NUM_WARPS * 32;      // 512

__global__ void wmma_bf16_gemm_abt(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   __nv_bfloat16* __restrict__ C,
                                   int M, int N, int K) {
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    __shared__ __nv_bfloat16 A_smem[BM][BK + PAD];
    __shared__ __nv_bfloat16 B_smem[BN][BK + PAD];   // 按 N x K 行主序存
    __shared__ float         C_smem[NUM_WARPS][16][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // -------- 协作加载 A tile: BM x BK --------
        for (int i = tid; i < BM * BK; i += THREADS) {
            int r = i / BK;
            int c = i % BK;
            int gr = block_m + r;
            int gc = k0 + c;
            A_smem[r][c] = (gr < M && gc < K) ? A[gr * K + gc]
                                              : __float2bfloat16(0.0f);
        }
        // -------- 协作加载 B tile: BN x BK --------
        for (int i = tid; i < BN * BK; i += THREADS) {
            int r = i / BK;
            int c = i % BK;
            int gr = block_n + r;
            int gc = k0 + c;
            B_smem[r][c] = (gr < N && gc < K) ? B[gr * K + gc]
                                              : __float2bfloat16(0.0f);
        }
        __syncthreads();

       for (int kk = 0; kk < BK; kk += 16) {
            // A 片段：从 shared 的 kk 列开始，取 16 列
            wmma::load_matrix_sync(a_frag,
                               &A_smem[warp_m * 16][kk],
                               BK + PAD);

            // B 片段：从 shared 的 kk 列开始，取 16 列，col_major 实现 B^T
            wmma::load_matrix_sync(b_frag,
                               &B_smem[warp_n * 16][kk],
                               BK + PAD);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();   // 下一轮覆盖 shared memory 前同步
    }

    // -------- 结果写回 --------
    wmma::store_matrix_sync(&C_smem[warp_id][0][0], c_frag, 16,
                            wmma::mem_row_major);

    const int tile_m = block_m + warp_m * 16;
    const int tile_n = block_n + warp_n * 16;

    for (int i = lane; i < 16 * 16; i += 32) {
        int r = i / 16;
        int c = i % 16;
        int gr = tile_m + r;
        int gc = tile_n + c;
        if (gr < M && gc < N) {
            C[gr * N + gc] = __float2bfloat16(C_smem[warp_id][r][c]);
        }
    }
}

extern "C" void run_kernel(const __nv_bfloat16* A,
                           const __nv_bfloat16* B,
                           __nv_bfloat16* C,
                           int M, int N, int K) {
    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    wmma_bf16_gemm_abt<<<grid, block>>>(A, B, C, M, N, K);
}