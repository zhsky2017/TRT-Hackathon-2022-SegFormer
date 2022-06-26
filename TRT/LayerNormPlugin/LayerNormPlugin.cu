#include "LayerNormPlugin.h"
#include "layer_norm.cuh"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

// __global__ void layerNormKernel(float *pInput, float *pOutput)
// {
//     const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

//     __shared__ float temp[128];

//     float value0 = pInput[index];
//     float value1 = pInput[index + 128];

//     temp[tx] = value0 + value1;
//     __syncthreads();

//     for (int stride = 64; stride >= 1; stride /= 2)
//     {
//         if (tx < stride)
//         {
//             temp[tx] += temp[tx + stride];
//         }
//         __syncthreads();
//     }
//     float mean = temp[0] / 256;
//     __syncthreads();

//     temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
//     __syncthreads();

//     for (int stride = 64; stride >= 1; stride /= 2)
//     {
//         if (tx < stride)
//         {
//             temp[tx] += temp[tx + stride];
//         }
//         __syncthreads();
//     }
//     float var = temp[0] / 256;

//     pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6);
//     pOutput[index + 128] = (value1 - mean) * rsqrtf(var + 6e-6);
// }

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    oneflow::cuda::layer_norm::DirectLoad<float, float>  load  = {(float *)inputs[0],  inputDesc[0].dims.d[2]};
    oneflow::cuda::layer_norm::DirectStore<float, float> store = {(float *)outputs[0], inputDesc[0].dims.d[2]};
    double epsilon = 6e-6;
    float * mean, * inv_variance;
    cudaMalloc((void **)&mean, sizeof(float));
    cudaMalloc((void **)&inv_variance, sizeof(float));
    switch(int64_t(inputDesc[0].dims.d[2] / 32)) {
        case 2:
            // std::cout << inputDesc[0].dims.d[2] / 32 << std::endl;
            // std::cout << "nBlock: "<< nBlock << std::endl;
            oneflow::cuda::layer_norm::LaunchLayerNormWarpImpl<oneflow::cuda::layer_norm::DirectLoad<float, float>, oneflow::cuda::layer_norm::DirectStore<float, float>, float, 2, 2, 32, 4, false>
            (stream, load, store, int64_t(nBlock), int64_t(inputDesc[0].dims.d[2]), epsilon, mean, inv_variance);
            break;
        case 4:
            // std::cout << inputDesc[0].dims.d[2] / 32 << std::endl;
            // std::cout << "nBlock: "<< nBlock << std::endl;
            oneflow::cuda::layer_norm::LaunchLayerNormWarpImpl<oneflow::cuda::layer_norm::DirectLoad<float, float>, oneflow::cuda::layer_norm::DirectStore<float, float>, float, 4, 4, 32, 2, false>
            (stream, load, store, int64_t(nBlock), int64_t(inputDesc[0].dims.d[2]), epsilon, mean, inv_variance);
            break;

        case 8:
            // std::cout << inputDesc[0].dims.d[2] / 32 << std::endl;
            // std::cout << "nBlock: "<< nBlock << std::endl;
            oneflow::cuda::layer_norm::LaunchLayerNormWarpImpl<oneflow::cuda::layer_norm::DirectLoad<float, float>, oneflow::cuda::layer_norm::DirectStore<float, float>, float, 4, 8, 32, 1, false>
            (stream, load, store, int64_t(nBlock), int64_t(inputDesc[0].dims.d[2]), epsilon, mean, inv_variance);
            break;
        case 10:
            // std::cout << inputDesc[0].dims.d[2] / 32 << std::endl;
            // std::cout << "nBlock: "<< nBlock << std::endl;
            oneflow::cuda::layer_norm::LaunchLayerNormWarpImpl<oneflow::cuda::layer_norm::DirectLoad<float, float>, oneflow::cuda::layer_norm::DirectStore<float, float>, float, 2, 10, 32, 1, false>
            (stream, load, store, int64_t(nBlock), int64_t(inputDesc[0].dims.d[2]), epsilon, mean, inv_variance);
            break;

        case 16:
            // std::cout << inputDesc[0].dims.d[2] / 32 << std::endl;
            // std::cout << "nBlock: "<< nBlock << std::endl;
            oneflow::cuda::layer_norm::LaunchLayerNormWarpImpl<oneflow::cuda::layer_norm::DirectLoad<float, float>, oneflow::cuda::layer_norm::DirectStore<float, float>, float, 4, 16, 32, 1, false>
            (stream, load, store, int64_t(nBlock), int64_t(inputDesc[0].dims.d[2]), epsilon, mean, inv_variance);
            break;
        
    }
    cudaFree(mean);
    cudaFree(inv_variance);
    // layerNormKernel <<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

