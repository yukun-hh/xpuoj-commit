#include <stdint.h>
#include <cuda_fp16.h>
__global__ void vector_add(__half* A, const __half* B, int N) {
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N) A[idx]=__hadd(A[idx],B[idx]);
}
extern "C" void run_kernel(__half* A, const __half* B, int64_t numel){
    int threadsPerBlock = 256;
    int blocksPerGrid = (numel + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, numel);
    cudaDeviceSynchronize();
}
