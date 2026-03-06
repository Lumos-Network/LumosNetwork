#include "softmax_gpu.h"

__global__ void softmax_kernel(float *data, int num, float *space, float *ALPHA)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = data[index] / ALPHA[0];
}

__global__ void softmax_gradient_kernel(float *data, int num, float *space, float *ALPHA)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = (data[index] + ALPHA[0]) * data[index];
}

void softmax_gpu(float *data, int num, float *space, float *ALPHA)
{
    softmax_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space, ALPHA);
}

void softmax_gradient_gpu(float *data, int num, float *space, float *ALPHA)
{
    softmax_gradient_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, space, ALPHA);
}

void softmax_exp_sum_gpu(float *data, int num, float *workspace, float *space)
{
    max_gpu(data, num, workspace+num);
    exp_list_gpu(data, num, workspace, workspace+num);
    sum_gpu(workspace, num, space);
}

__global__ void log_softmax_kernel(float *data, int num, float *max, float *space, float *ALPHA)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = (data[index] - max[0]) - logf(ALPHA[0]);
}

__global__ void log_softmax_gradient_kernel(float *data, int num, float *max, float *space, float *ALPHA)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = 1 - exp(data[index] - max[0]) / ALPHA[0];
}

void log_softmax_gpu(float *data, int num, float *space, float *ALPHA)
{
    float *max;
    cudaMalloc((void**)&max, sizeof(float));
    max_gpu(data, num, max);
    log_softmax_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, max, space, ALPHA);
    cudaFree(max);
}

void log_softmax_gradient_gpu(float *data, int num, float *space, float *ALPHA)
{
    float *max;
    cudaMalloc((void**)&max, sizeof(float));
    max_gpu(data, num, max);
    log_softmax_gradient_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, num, max, space, ALPHA);
    cudaFree(max);
}
