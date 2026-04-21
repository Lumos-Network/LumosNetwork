#include "bias_gpu.h"

__global__ void add_bias_kernel(float *origin, float *bias, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    origin[(k*n+j)*size + i] += bias[j];
}

__global__ void scale_bias_kernel(float *origin, float *bias, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    origin[(k*n+j)*size + i] *= bias[j];
}

void add_bias_gpu(float *origin, float *bias, int batch, int n, int size)
{
    size_t num = n*size*batch;
    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(origin, bias, batch, n, size);
}

void scale_bias_gpu(float *origin, float *bias, int batch, int n, int size)
{
    size_t num = n*size*batch;
    scale_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(origin, bias, batch, n, size);
}

__global__ void backward_bias_conn_kernel(float *bias_delta, float *delta, int batch, int n)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int b;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    bias_delta[index] += sum;
}

__global__ void backward_bias_kernel(float *bias_delta, float *delta, int batch, int n, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_delta[filter] += part[i];
    }
}

void backward_bias_gpu(float *bias_delta, float *delta, int batch, int n, int size)
{
    if(size == 1){
        backward_bias_conn_kernel<<<cuda_gridsize(size_t(n)), BLOCK>>>(bias_delta, delta, batch, n);
    }else{
        backward_bias_kernel<<<n, BLOCK>>>(bias_delta, delta, batch, n, size);
    }
}
