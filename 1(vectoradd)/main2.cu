#include <stdint.h>
#include <cuda_fp16.h>
__global__ void vector_add(__half* A, const __half* B, int numel) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    __half2* A2 = reinterpret_cast< __half2* >(A);
    const __half2* B2 = reinterpret_cast<const __half2* >(B);
    int base = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int i0 = base;
    int i1 = base + blockDim.x;
    int i2 = base + blockDim.x * 2;
    int i3 = base + blockDim.x * 3;
    int n2 = numel/2;
    if(i0 < n2) A2[i0] = __hadd2(A2[i0],B2[i0]);
    if(i1 < n2) A2[i1] = __hadd2(A2[i1],B2[i1]);
    if(i2 < n2) A2[i2] = __hadd2(A2[i2],B2[i2]);
    if(i3 < n2) A2[i3] = __hadd2(A2[i3],B2[i3]);
    if(idx == 0 && (numel&1)){
        A[numel-1]= __hadd(A[numel-1],B[numel-1]);
    }
}
extern "C" void run_kernel(__half* A, const __half* B, int64_t numel){
    int threads = 1024;
    int numel2=numel/2;
    int works=(numel2+3)/4;
    int blocks = (works + threads-1) / threads;
    vector_add<<<blocks,threads>>>(A,B,numel);
}
