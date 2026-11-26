// Single-precision matrix multiply, C = A * B, in six stages.
// A is M x K, B is K x N, C is M x N, all row-major, fp32.
// Every kernel handles any M, N, K (no multiple-of-tile requirement).

#include <cuda_runtime.h>
#include <mma.h>

#include "gkl.h"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------------------
// Stage 0: naive. One thread per output element. threadIdx.x walks rows, so
// neighbouring threads in a warp read A rows K floats apart and write C
// elements N floats apart: every global access is uncoalesced.
// ---------------------------------------------------------------------------
__global__ void matmul_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ---------------------------------------------------------------------------
// Stage 1: coalesced. Same arithmetic, but threadIdx.x walks columns. A warp
// now reads one A element (broadcast) and 32 consecutive B elements per k.
// ---------------------------------------------------------------------------
__global__ void matmul_coalesced(int M, int N, int K, const float* A, const float* B, float* C) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: shared-memory tiling. A 32x32 block cooperatively stages a 32x32
// tile of A and of B in shared memory, then each thread reuses each loaded
// value 32 times. Global traffic drops by the tile width.
// ---------------------------------------------------------------------------
constexpr int SMEM_TILE = 32;

__global__ void matmul_smem(int M, int N, int K, const float* A, const float* B, float* C) {
  __shared__ float As[SMEM_TILE][SMEM_TILE];
  __shared__ float Bs[SMEM_TILE][SMEM_TILE];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * SMEM_TILE + ty;
  int col = blockIdx.x * SMEM_TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += SMEM_TILE) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < SMEM_TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Stage 3: 2D register blocking. A block of 256 threads computes a 128x128
// tile of C; each thread owns an 8x8 sub-tile held in registers. Per k step a
// thread loads 8 A values and 8 B values from shared memory into registers
// and does 64 FMAs, so shared-memory loads per FMA drop from 2 to 1/4.
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int RB_THREADS = (BM / TM) * (BN / TN);  // 256

__global__ void __launch_bounds__(RB_THREADS)
matmul_regblock(int M, int N, int K, const float* A, const float* B, float* C) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  const int tid = threadIdx.x;
  const int trow = tid / (BN / TN);  // 0..15
  const int tcol = tid % (BN / TN);  // 0..15
  const int brow = blockIdx.y * BM, bcol = blockIdx.x * BN;

  float acc[TM][TN] = {};
  float a_reg[TM], b_reg[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // 128*8 = 1024 elements per tile, 4 per thread.
    for (int i = tid; i < BM * BK; i += RB_THREADS) {
      int r = i / BK, c = i % BK;
      int gr = brow + r, gc = k0 + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
    }
    for (int i = tid; i < BK * BN; i += RB_THREADS) {
      int r = i / BN, c = i % BN;
      int gr = k0 + r, gc = bcol + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int m = 0; m < TM; ++m) a_reg[m] = As[trow * TM + m][k];
#pragma unroll
      for (int n = 0; n < TN; ++n) b_reg[n] = Bs[k][tcol * TN + n];
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] += a_reg[m] * b_reg[n];
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int gr = brow + trow * TM + m;
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      int gc = bcol + tcol * TN + n;
      if (gr < M && gc < N) C[gr * N + gc] = acc[m][n];
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 4: vectorized. Stage 3 plus:
//  * 128-bit (float4) global loads of A and B and float4 stores of C,
//  * A stored transposed in shared memory (AsT[k][m]) so the inner loop reads
//    a thread's 8 A values as two float4s instead of 8 strided scalars.
// float4 global access needs 16-byte alignment, so the vector path runs only
// when K and N are multiples of 4; otherwise the kernel takes scalar loads.
// The flag is uniform across the grid, so the branch never diverges.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(RB_THREADS)
matmul_vectorized(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
                  float* __restrict__ C) {
  __shared__ __align__(16) float AsT[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  const bool vec = (K % 4 == 0) && (N % 4 == 0);
  const int tid = threadIdx.x;
  const int trow = tid / (BN / TN);
  const int tcol = tid % (BN / TN);
  const int brow = blockIdx.y * BM, bcol = blockIdx.x * BN;

  float acc[TM][TN] = {};
  float a_reg[TM], b_reg[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    if (vec) {
      // A tile: 128 rows x 8 cols = 256 float4, one per thread.
      {
        int r = tid / (BK / 4), c = (tid % (BK / 4)) * 4;
        int gr = brow + r, gc = k0 + c;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gr < M && gc < K) v = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
        AsT[c + 0][r] = v.x;
        AsT[c + 1][r] = v.y;
        AsT[c + 2][r] = v.z;
        AsT[c + 3][r] = v.w;
      }
      // B tile: 8 rows x 128 cols = 256 float4, one per thread.
      {
        int r = tid / (BN / 4), c = (tid % (BN / 4)) * 4;
        int gr = k0 + r, gc = bcol + c;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gr < K && gc < N) v = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
        *reinterpret_cast<float4*>(&Bs[r][c]) = v;
      }
    } else {
      for (int i = tid; i < BM * BK; i += RB_THREADS) {
        int r = i / BK, c = i % BK;
        int gr = brow + r, gc = k0 + c;
        AsT[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
      }
      for (int i = tid; i < BK * BN; i += RB_THREADS) {
        int r = i / BN, c = i % BN;
        int gr = k0 + r, gc = bcol + c;
        Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float4 a0 = *reinterpret_cast<const float4*>(&AsT[k][trow * TM]);
      float4 a1 = *reinterpret_cast<const float4*>(&AsT[k][trow * TM + 4]);
      float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tcol * TN]);
      float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][tcol * TN + 4]);
      a_reg[0] = a0.x; a_reg[1] = a0.y; a_reg[2] = a0.z; a_reg[3] = a0.w;
      a_reg[4] = a1.x; a_reg[5] = a1.y; a_reg[6] = a1.z; a_reg[7] = a1.w;
      b_reg[0] = b0.x; b_reg[1] = b0.y; b_reg[2] = b0.z; b_reg[3] = b0.w;
      b_reg[4] = b1.x; b_reg[5] = b1.y; b_reg[6] = b1.z; b_reg[7] = b1.w;
#pragma unroll
      for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] += a_reg[m] * b_reg[n];
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int gr = brow + trow * TM + m;
    if (gr >= M) continue;
    if (vec) {
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        int gc = bcol + tcol * TN + n;
        if (gc < N)
          *reinterpret_cast<float4*>(&C[gr * N + gc]) =
              make_float4(acc[m][n], acc[m][n + 1], acc[m][n + 2], acc[m][n + 3]);
      }
    } else {
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        int gc = bcol + tcol * TN + n;
        if (gc < N) C[gr * N + gc] = acc[m][n];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 5: tensor cores (TF32 WMMA, sm_80+). Inputs are rounded to TF32
// (10-bit mantissa) and multiplied on tensor cores with fp32 accumulation,
// the same contract as cuBLAS with allow_tf32=True. A block of 4 warps
// computes a 64x64 C tile; each warp owns a 32x32 piece as 2x2 fragments of
// 16x16. K advances 32 at a time through shared memory.
// ---------------------------------------------------------------------------
constexpr int WM = 64, WN = 64, WK = 32;
constexpr int WMMA_THREADS = 128;
constexpr int SKEW = 4;  // pad rows so warps do not hit the same smem bank

__global__ void __launch_bounds__(WMMA_THREADS)
matmul_wmma_tf32(int M, int N, int K, const float* A, const float* B, float* C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  __shared__ __align__(32) float As[WM][WK + SKEW];
  __shared__ __align__(32) float Bs[WK][WN + SKEW];
  __shared__ __align__(32) float Cs[WM][WN + SKEW];
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int wr = (warp / 2) * 32, wc = (warp % 2) * 32;
  const int brow = blockIdx.y * WM, bcol = blockIdx.x * WN;

  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  for (int k0 = 0; k0 < K; k0 += WK) {
    for (int i = tid; i < WM * WK; i += WMMA_THREADS) {
      int r = i / WK, c = i % WK;
      int gr = brow + r, gc = k0 + c;
      As[r][c] = wmma::__float_to_tf32((gr < M && gc < K) ? A[gr * K + gc] : 0.f);
    }
    for (int i = tid; i < WK * WN; i += WMMA_THREADS) {
      int r = i / WN, c = i % WN;
      int gr = k0 + r, gc = bcol + c;
      Bs[r][c] = wmma::__float_to_tf32((gr < K && gc < N) ? B[gr * N + gc] : 0.f);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < WK; kk += 8) {
      wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[wr + i * 16][kk], WK + SKEW);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[kk][wc + j * 16], WN + SKEW);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  // Stage through shared memory so edge tiles can be stored with bounds checks.
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(&Cs[wr + i * 16][wc + j * 16], acc[i][j], WN + SKEW, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < WM * WN; i += WMMA_THREADS) {
    int r = i / WN, c = i % WN;
    int gr = brow + r, gc = bcol + c;
    if (gr < M && gc < N) C[gr * N + gc] = Cs[r][c];
  }
#endif
}

extern "C" int gkl_matmul(int stage, int M, int N, int K, const float* A, const float* B, float* C,
                          cudaStream_t stream) {
  switch (stage) {
    case 0: {
      dim3 block(32, 32), grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
      matmul_naive<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    case 1: {
      dim3 block(32, 32), grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
      matmul_coalesced<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    case 2: {
      dim3 block(SMEM_TILE, SMEM_TILE), grid(CEIL_DIV(N, SMEM_TILE), CEIL_DIV(M, SMEM_TILE));
      matmul_smem<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    case 3: {
      dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      matmul_regblock<<<grid, RB_THREADS, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    case 4: {
      dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      matmul_vectorized<<<grid, RB_THREADS, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    case 5: {
      int dev = 0, major = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
      if (major < 8) return GKL_ERR_UNSUPPORTED;
      dim3 grid(CEIL_DIV(N, WN), CEIL_DIV(M, WM));
      matmul_wmma_tf32<<<grid, WMMA_THREADS, 0, stream>>>(M, N, K, A, B, C);
      break;
    }
    default:
      return GKL_ERR_BAD_STAGE;
  }
  return static_cast<int>(cudaGetLastError());
}
