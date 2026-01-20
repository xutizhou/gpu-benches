#include "../MeasurementSeries.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Use 32-bit accesses to match the shared memory bank width (4 bytes).
using smem_t = uint32_t;

namespace {

constexpr int kMaxBlockSize = 1024;
constexpr int kNumArrays = 4;  // A, B, C, D
constexpr size_t kBaseShmemBytes =
    static_cast<size_t>(kNumArrays) * kMaxBlockSize * sizeof(smem_t);

using kernel_ptr_type = void (*)(smem_t* out, int iters);

__device__ __forceinline__ smem_t mix_u32(smem_t x) {
  // A tiny mixing function to make values non-trivial (prevents constant folding).
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__global__ void shmem_init_kernel(smem_t* out, int iters) {
  extern __shared__ smem_t smem_raw[];
  volatile smem_t* A = smem_raw + 0 * kMaxBlockSize;
  volatile smem_t* B = smem_raw + 1 * kMaxBlockSize;
  volatile smem_t* C = smem_raw + 2 * kMaxBlockSize;
  volatile smem_t* D = smem_raw + 3 * kMaxBlockSize;

  const int tid = threadIdx.x;
  const smem_t base = mix_u32(static_cast<smem_t>(tid + 1));

  // Initialize sources (not counted in bandwidth model; outside the timed loop).
  B[tid] = base;
  C[tid] = base ^ 0x12345678U;
  D[tid] = base ^ 0xdeadbeefu;
  __syncthreads();

  // STREAM "init": 1 shared store stream.
  const smem_t v = base + 3;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    A[tid] = v;
  }

  if (tid == 0) {
    out[blockIdx.x] = static_cast<smem_t>(A[0]);
  }
}

__global__ void shmem_read_kernel(smem_t* out, int iters) {
  extern __shared__ smem_t smem_raw[];
  volatile smem_t* A = smem_raw + 0 * kMaxBlockSize;

  const int tid = threadIdx.x;
  A[tid] = mix_u32(static_cast<smem_t>(tid + 1));
  __syncthreads();

  // STREAM "read": 1 shared load stream.
  smem_t acc = 0;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    acc += A[tid];
  }

  if (tid == 0) {
    out[blockIdx.x] = acc;
  }
}

__global__ void shmem_copy_kernel(smem_t* out, int iters) {
  extern __shared__ smem_t smem_raw[];
  volatile smem_t* A = smem_raw + 0 * kMaxBlockSize;
  volatile smem_t* B = smem_raw + 1 * kMaxBlockSize;

  const int tid = threadIdx.x;
  const int peer = tid ^ 1;  // avoid store->load to same address patterns

  B[tid] = mix_u32(static_cast<smem_t>(tid + 1));
  __syncthreads();

  smem_t acc = 0;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    const smem_t x = B[peer];
    A[tid] = x;
    acc += x;
  }

  if (tid == 0) {
    out[blockIdx.x] = acc;
  }
}

__global__ void shmem_scale_kernel(smem_t* out, int iters) {
  extern __shared__ smem_t smem_raw[];
  volatile smem_t* A = smem_raw + 0 * kMaxBlockSize;
  volatile smem_t* B = smem_raw + 1 * kMaxBlockSize;

  const int tid = threadIdx.x;
  const int peer = tid ^ 1;

  B[tid] = mix_u32(static_cast<smem_t>(tid + 1));
  __syncthreads();

  const smem_t scalar = 3;
  smem_t acc = 0;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    const smem_t x = B[peer];
    const smem_t y = x * scalar;
    A[tid] = y;
    acc += y;
  }

  if (tid == 0) {
    out[blockIdx.x] = acc;
  }
}

__global__ void shmem_add_kernel(smem_t* out, int iters) {
  extern __shared__ smem_t smem_raw[];
  volatile smem_t* A = smem_raw + 0 * kMaxBlockSize;
  volatile smem_t* B = smem_raw + 1 * kMaxBlockSize;
  volatile smem_t* C = smem_raw + 2 * kMaxBlockSize;

  const int tid = threadIdx.x;
  const int peer = tid ^ 1;

  const smem_t base = mix_u32(static_cast<smem_t>(tid + 1));
  B[tid] = base;
  C[tid] = base ^ 0x9e3779b9U;
  __syncthreads();

  smem_t acc = 0;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    const smem_t x = B[peer];
    const smem_t y = C[tid];
    const smem_t z = x + y;
    A[tid] = z;
    acc += z;
  }

  if (tid == 0) {
    out[blockIdx.x] = acc;
  }
}

__global__ void shmem_triad_kernel(smem_t* out, int iters) {
  extern __shared__ smem_t smem_raw[];
  volatile smem_t* A = smem_raw + 0 * kMaxBlockSize;
  volatile smem_t* B = smem_raw + 1 * kMaxBlockSize;
  volatile smem_t* C = smem_raw + 2 * kMaxBlockSize;
  volatile smem_t* D = smem_raw + 3 * kMaxBlockSize;

  const int tid = threadIdx.x;
  const int peer = tid ^ 1;

  const smem_t base = mix_u32(static_cast<smem_t>(tid + 1));
  B[tid] = base;
  C[tid] = base ^ 0x7f4a7c15U;
  D[tid] = base ^ 0xf39cc060U;
  __syncthreads();

  smem_t acc = 0;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    const smem_t x = B[peer];
    const smem_t y = C[tid];
    const smem_t z = D[peer];
    const smem_t r = x + y * z;
    A[tid] = r;
    acc += r;
  }

  if (tid == 0) {
    out[blockIdx.x] = acc;
  }
}

struct KernelDesc {
  const char* name;
  kernel_ptr_type func;
  int streamCount;  // number of memory streams (load/store) per element
};

struct Result {
  double tb_s;
  double B_per_cycle_per_SM;
};

static Result measureKernel(const KernelDesc& k,
                            int blockSize,
                            int blocksPerSM,
                            int iters,
                            int runs,
                            double sm_clock_mhz) {
  cudaDeviceProp prop{};
  int deviceId = 0;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));

  // Guard invalid configurations.
  if (blockSize <= 0 || blockSize % 32 != 0 || blockSize > prop.maxThreadsPerBlock)
    return {0.0, 0.0};
  if (blockSize * blocksPerSM > prop.maxThreadsPerMultiProcessor)
    return {0.0, 0.0};

  // Make sure dynamic shared memory is allowed if we ever raise it above the
  // default per-block limit.
  if (kBaseShmemBytes > prop.sharedMemPerBlock) {
    GPU_ERROR(cudaFuncSetAttribute(
        k.func, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kBaseShmemBytes)));
  }

  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, k.func, blockSize, static_cast<int>(kBaseShmemBytes)));
  if (maxActiveBlocks < blocksPerSM) {
    return {0.0, 0.0};
  }

  const int smCount = prop.multiProcessorCount;
  const int blockCount = std::max(1, smCount * blocksPerSM);

  smem_t* dOut = nullptr;
  GPU_ERROR(cudaMalloc(&dOut, static_cast<size_t>(blockCount) * sizeof(smem_t)));

  // Warmup.
  k.func<<<blockCount, blockSize, kBaseShmemBytes>>>(dOut, iters);
  GPU_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  MeasurementSeries times;
  for (int r = 0; r < runs; ++r) {
    GPU_ERROR(cudaEventRecord(start));
    k.func<<<blockCount, blockSize, kBaseShmemBytes>>>(dOut, iters);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    float ms = 0.0f;
    GPU_ERROR(cudaEventElapsedTime(&ms, start, stop));
    times.add(static_cast<double>(ms) / 1e3);
  }

  GPU_ERROR(cudaGetLastError());

  const double dt = times.median();  // STREAM-style: use median-of-N
  const double bytes =
      static_cast<double>(k.streamCount) * static_cast<double>(blockCount) *
      static_cast<double>(blockSize) * static_cast<double>(iters) *
      static_cast<double>(sizeof(smem_t));
  const double bw_TB_s = bytes / dt / 1.0e12;

  const double cycles = dt * sm_clock_mhz * 1.0e6;
  const double B_per_cycle_per_SM = bytes / cycles / static_cast<double>(smCount);

  GPU_ERROR(cudaFree(dOut));
  GPU_ERROR(cudaEventDestroy(start));
  GPU_ERROR(cudaEventDestroy(stop));
  return {bw_TB_s, B_per_cycle_per_SM};
}

}  // namespace

int main(int argc, char** argv) {
  int iters = 200000;
  int runs = 11;
  int blocksPerSM = 2;

  if (argc >= 2)
    iters = std::atoi(argv[1]);
  if (argc >= 3)
    runs = std::atoi(argv[2]);
  if (argc >= 4)
    blocksPerSM = std::atoi(argv[3]);

  cudaDeviceProp prop{};
  int deviceId = 0;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));

  cout << "GPU: " << prop.name << "  SMs: " << prop.multiProcessorCount
       << "  CC: " << prop.major << "." << prop.minor << "\n";
  cout << "iters=" << iters << " runs=" << runs << " blocksPerSM=" << blocksPerSM
       << "\n";

  // Lock the clock to a steady value (prints progress while ramping up).
  const double sm_clock_mhz = static_cast<double>(getGPUClock());
  cout << "SM clock (NVML): " << fixed << setprecision(0) << sm_clock_mhz
       << " MHz\n";
  cout << "shared per block (base): " << (kBaseShmemBytes / 1024.0) << " KB\n\n";

  const vector<KernelDesc> kernels = {
      {"init", shmem_init_kernel, 1},   // 1 store
      {"read", shmem_read_kernel, 1},   // 1 load
      {"copy", shmem_copy_kernel, 2},   // 1 load + 1 store
      {"scale", shmem_scale_kernel, 2}, // 1 load + 1 store
      {"add", shmem_add_kernel, 3},     // 2 load + 1 store
      {"triad", shmem_triad_kernel, 4}, // 3 load + 1 store
  };

  cout << "block smBlocks   threads   occ%   |";
  for (const auto& k : kernels) {
    cout << setw(9) << k.name;
  }
  cout << "    (TB/s)\n";

  for (int blockSize = 32; blockSize <= 1024; blockSize += 32) {
    // Skip invalid occupancy configs (STREAM-style sweep).
    if (blockSize * blocksPerSM > prop.maxThreadsPerMultiProcessor ||
        blockSize > prop.maxThreadsPerBlock) {
      continue;
    }

    const int smCount = prop.multiProcessorCount;
    const int threads = smCount * blockSize * blocksPerSM;
    const double occ_pct =
        100.0 * static_cast<double>(blockSize * blocksPerSM) /
        static_cast<double>(prop.maxThreadsPerMultiProcessor);

    cout << setw(4) << blockSize << "   " << setw(6) << blocksPerSM << "  "
         << setw(9) << threads << "  " << setw(5) << setprecision(1) << fixed
         << occ_pct << "%  |";

    for (const auto& k : kernels) {
      const Result r =
          measureKernel(k, blockSize, blocksPerSM, iters, runs, sm_clock_mhz);
      cout << setw(9) << setprecision(2) << fixed << r.tb_s;
    }
    cout << "\n";
  }
  return 0;
}
