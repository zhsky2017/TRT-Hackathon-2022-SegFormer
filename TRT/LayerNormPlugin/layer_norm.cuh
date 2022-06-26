#ifndef ONEFLOW_CORE_CUDA_LAYER_NORM_H_
#define ONEFLOW_CORE_CUDA_LAYER_NORM_H_

#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>

namespace oneflow {

namespace cuda {

namespace layer_norm {

constexpr int kWarpSize = 32;

template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) 
{
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask, thread_group_width));
    }
    return val;
}

template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) 
{
    typedef cub::BlockReduce<T, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
    if (threadIdx.x == 0) { result_broadcast = result; }
    __syncthreads();
    return result_broadcast;
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
    return __fdividef(a, b);
#else
    return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
    return a / b;
}

template<typename T>
__inline__ __device__ T Rsqrt(T x);

template<>
__inline__ __device__ float Rsqrt<float>(float x) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
    return __frsqrt_rn(x);
#else
    return rsqrt(x);
#endif
}

template<>
__inline__ __device__ double Rsqrt<double>(double x) {
    return rsqrt(x);
}

template<class Func>
inline cudaError_t GetNumBlocks(Func func, int64_t block_size, size_t dynamic_smem_size,
                                int64_t max_blocks, int64_t waves, int* num_blocks) 
{
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess) { return err; }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess) { return err; }
    }
    int max_active_blocks;
    {
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, func,
                                                                        block_size, dynamic_smem_size);
    }
    *num_blocks =
        std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
    return cudaSuccess;
}

template<typename T>
struct DefaultComputeType {
    using type = T;
};


template<typename T, int N>
struct GetPackType {
    using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
    static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
    __device__ Pack() {
        // do nothing
    }
    PackType<T, N> storage;
    T elem[N];
};

template<typename SRC, typename DST>
struct DirectLoad 
{
    DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
    template<int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) const {
        Pack<SRC, N> pack;
        const int64_t offset = (row * row_size + col) / N;
        pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
    #pragma unroll
        for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
    }
    const SRC* src;
    int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore 
{
    DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
    template<int N>
    __device__ void store(const SRC* src, int64_t row, int64_t col) {
        Pack<DST, N> pack;
        const int64_t offset = (row * row_size + col) / N;
    #pragma unroll
        for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
        *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
    }
    DST* dst;
    int64_t row_size;
};

template<typename T>
inline __device__ void WelfordCombine(T val, T* mean, T* m2, T* count) {
    // Use Welford Online algorithem to compute mean and variance
    // For more details you can refer to:
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    *count += 1;
    T delta1 = val - *mean;
    *mean += Div(delta1, *count);
    T delta2 = val - *mean;
    *m2 += delta1 * delta2;
    }

template<typename T>
inline __device__ void WelfordCombine(T b_mean, T b_m2, T b_count, T* mean, T* m2, T* count) 
{
    if (b_count == 0) { return; }
    T new_count = *count + b_count;
    T nb_over_n = Div(b_count, new_count);
    T delta = b_mean - *mean;
    *mean += delta * nb_over_n;
    *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
    *count = new_count;
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                             T* m2, T* count) 
{
    *mean = thread_mean;
    *m2 = thread_m2;
    *count = thread_count;
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
        T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
        T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
        WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
    }
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                                T* m2, T* count) 
{
    WelfordWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_count, mean, m2, count);
    *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
    *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
    *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

template<typename T>
__inline__ __device__ void WelfordBlockAllReduce(T thread_mean, T thread_m2, T thread_count,
                                                 T* result_mean, T* result_m2, T* result_count) {
    __shared__ T mean_shared[kWarpSize];
    __shared__ T m2_shared[kWarpSize];
    __shared__ T count_shared[kWarpSize];
    __shared__ T mean_result_broadcast;
    __shared__ T m2_result_broadcast;
    __shared__ T count_result_broadcast;
    const int lid = threadIdx.x % kWarpSize;
    const int wid = threadIdx.x / kWarpSize;
    T warp_mean = 0;
    T warp_m2 = 0;
    T warp_count = 0;
    WelfordWarpReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
    __syncthreads();
    if (lid == 0) {
        mean_shared[wid] = warp_mean;
        m2_shared[wid] = warp_m2;
        count_shared[wid] = warp_count;
    }
    __syncthreads();
    if (wid == 0) {
        if (threadIdx.x < blockDim.x / kWarpSize) {
        warp_mean = mean_shared[lid];
        warp_m2 = m2_shared[lid];
        warp_count = count_shared[lid];
        } else {
        warp_mean = static_cast<T>(0);
        warp_m2 = static_cast<T>(0);
        warp_count = static_cast<T>(0);
        }
        __syncwarp();
        T block_mean = 0;
        T block_m2 = 0;
        T block_count = 0;
        WelfordWarpReduce(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
        if (lid == 0) {
        mean_result_broadcast = block_mean;
        m2_result_broadcast = block_m2;
        count_result_broadcast = block_count;
        }
    }
    __syncthreads();
    *result_mean = mean_result_broadcast;
    *result_m2 = m2_result_broadcast;
    *result_count = count_result_broadcast;
}
// ------- template args -----------
// ComputeType:        float
// pack_size:          128 / 32 = 4
// cols_per_thread:    256 / 32 = 8
// thread_group_width: 32
// row_per_access:     1
// padding:            false
// ------- function args -----------
// rows: Dynamic
// cols: 256
// mean: return value
// inv_variance: return value   
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType* mean,
                                  ComputeType* inv_variance) 
{
    static_assert(cols_per_thread % pack_size == 0, "");
    static_assert(thread_group_width <= kWarpSize, "");
    static_assert(kWarpSize % thread_group_width == 0, "");
    constexpr int num_packs = cols_per_thread / pack_size;
    assert(cols <= cols_per_thread * thread_group_width);
    // buf[1][8]
    ComputeType buf[rows_per_access][cols_per_thread];
    // blockDim.x = 32, blockDim.y = 4
    // gridDim.x  = _
    const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int64_t num_global_thread_group = gridDim.x * blockDim.y;
    const int64_t lane_id = threadIdx.x;
    const int64_t step = num_global_thread_group * rows_per_access;
    // loop only once
    for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
        ComputeType thread_mean[rows_per_access];
        ComputeType thread_m2[rows_per_access];
        ComputeType thread_count[rows_per_access];
        #pragma unroll
        // loop only once
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            thread_mean[row_id] = 0;
            thread_m2[row_id] = 0;
            thread_count[row_id] = 0;
            ComputeType* row_buf = buf[row_id];
            #pragma unroll
            // num_packs = 2; loop twice
            for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
                const int col = (pack_id * thread_group_width + lane_id) * pack_size;
                const int pack_offset = pack_id * pack_size;
                if (!padding || col < cols) {
                    load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
                    #pragma unroll
                    for (int i = 0; i < pack_size; ++i) {
                        WelfordCombine(row_buf[pack_offset + i], thread_mean + row_id, thread_m2 + row_id,
                                    thread_count + row_id);
                    }
                } 
                else {
                #pragma unroll
                for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = 0; }
                }
            }
        }
        // 每个thread计算了8个元素的均值、方差(👆🏻)
        // 每个warp计算了256个元素的均值、方差(👇🏻)
        ComputeType warp_mean[rows_per_access];
        ComputeType warp_m2[rows_per_access];
        ComputeType warp_count[rows_per_access];
        #pragma unroll
        // loop only once
        for (int row_id = 0; row_id < rows_per_access; ++row_id) {
            int global_row_id = row + row_id;
            ComputeType* row_buf = buf[row_id];
            WelfordWarpAllReduce<ComputeType, thread_group_width>(
                thread_mean[row_id], thread_m2[row_id], thread_count[row_id], warp_mean + row_id,
                warp_m2 + row_id, warp_count + row_id);
            ComputeType row_mean = warp_mean[row_id];
            ComputeType row_variance =
                max(Div(warp_m2[row_id], warp_count[row_id]), static_cast<ComputeType>(0.0));
            ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
            if (lane_id == 0) {
                mean[global_row_id] = row_mean;
                inv_variance[global_row_id] = row_inv_var;
            }
            #pragma unroll
            for (int i = 0; i < cols_per_thread; ++i) {
                row_buf[i] = (row_buf[i] - row_mean) * row_inv_var;
            }
            #pragma unroll
            for (int i = 0; i < num_packs; ++i) {
                const int col = (i * thread_group_width + lane_id) * pack_size;
                if (!padding || col < cols) {
                store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col);
                }
            }
        }
    }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
inline cudaError_t LaunchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols,
                                           const double epsilon, ComputeType* mean,
                                           ComputeType* inv_variance) {
    constexpr int block_size = 128;
    constexpr int waves = 32;
    static_assert(block_size % thread_group_width == 0, "");
    constexpr int thread_groups_per_block = block_size / thread_group_width;
    dim3 block_dim(thread_group_width, thread_groups_per_block);
    const int64_t num_blocks =
        (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
    int grid_dim_x;
    {
        cudaError_t err =
            GetNumBlocks(LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                        thread_group_width, rows_per_access, padding>,
                        block_size, 0, num_blocks, waves, &grid_dim_x);
        if (err != cudaSuccess) { return err; }
    }
    LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                        rows_per_access, padding>
        <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
    return cudaPeekAtLastError();
}

} // layer_norm
 
} // cuda

} // oneflow

#endif // ONEFLOW_CORE_CUDA_LAYER_NORM_H_