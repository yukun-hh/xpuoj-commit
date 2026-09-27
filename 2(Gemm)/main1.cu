#include <stdint.h>
#include <cuda_bf16.h>

#define TILE 16
__global__ void MatMulShareBf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                                int64_t M, int64_t N, int64_t K) {
    __shared__ __nv_bfloat16 As[TILE][TILE];
    __shared__ __nv_bfloat16 Bs[TILE][TILE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty; // 对应输出 C 的 M 维度 (i)
    int col = blockIdx.x * TILE + tx; // 对应输出 C 的 N 维度 (j)
    
    float sum = 0.0f; // 使用 FP32 进行累加，保证计算精度
    
    int num = (K + TILE - 1) / TILE;
    
    for (int t = 0; t < num; t++) {
        int aCol = tx + t * TILE;
        int bK   = tx + t * TILE;
        int bN   = blockIdx.x * TILE + ty;   // 注意这里用 ty 作为 N 方向

        // A: A[row][k]
        As[ty][tx] = (row < M && aCol < K)
                        ? A[row * K + aCol]
                                : __float2bfloat16(0.0f);

        // B: B 是 (N,K)，这里存成 Bs[K][N]
        Bs[tx][ty] = (bN < N && bK < K)
                                ? B[bN * K + bK]
                                : __float2bfloat16(0.0f);

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            float a_val = __bfloat162float(As[ty][k]);
            float b_val = __bfloat162float(Bs[k][tx]); // Bs[K][N]
            sum += a_val * b_val;
        }
        
        __syncthreads();
    }
    
    // 将 FP32 结果转换回 bf16 写回全局内存
    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(sum);
    }
}
extern "C" void run_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int64_t M,
    int64_t N,
    int64_t K
){
    dim3 block16(TILE, TILE);
    dim3 grid16((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    MatMulShareBf16<<<grid16, block16>>>(A, B, C, M, N, K);
}