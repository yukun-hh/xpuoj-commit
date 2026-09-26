#include <stdint.h>
#include <cuda_fp16.h>
struct __align__(16) half4 {
    __half2 a, b, c, d;
};
__global__ void __launch_bounds__(256, 2) fp16_add_kernel(__half* __restrict__ A,
                const __half* __restrict__ B,
                int64_t numel) {
    int64_t idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) << 3;

    if (idx + 7 < numel) {
        half4 a4 = *reinterpret_cast<half4*>(A + idx);
        half4 b4 = *reinterpret_cast<const half4*>(B + idx);

        a4.a = __hadd2(a4.a, b4.a);
        a4.b = __hadd2(a4.b, b4.b);
        a4.c = __hadd2(a4.c, b4.c);
        a4.d = __hadd2(a4.d, b4.d);

        *reinterpret_cast<half4*>(A + idx) = a4;
    } else if (idx < numel) {
        #pragma unroll
        for (int i = 0; i < 8 && idx + i < numel; ++i)
            A[idx + i] = __hadd(A[idx + i], B[idx + i]);
    }
}

extern "C" void run_kernel(__half* A, const __half* B, int64_t numel) {

    const int threads = 256;
    int64_t blocks = (numel + ((int64_t)threads << 3) - 1) >> 11;

    fp16_add_kernel<<<blocks, threads>>>(A, B, numel);
}
