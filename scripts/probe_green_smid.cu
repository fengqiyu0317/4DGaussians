// Diagnostic kernel: record the physical SM of each CTA. The short bounded
// clock loop gives the scheduler enough work to visit each partition's SMs.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void record_smid(int* ids, unsigned long long cycles) {
    unsigned int sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    if (threadIdx.x == 0) ids[blockIdx.x] = int(sm);
    const unsigned long long start = clock64();
    while (clock64() - start < cycles) {
        asm volatile("");
    }
}

extern "C" int probe_launch_smid(uintptr_t stream, int* ids, unsigned int blocks) {
    record_smid<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(ids, 20000);
    return int(cudaGetLastError());
}
