#include "normalize_gpu.h"

__global__ void  fast_mean_kernel(float *x, int batch, int filters, int num, float *mean)
{
    const int threads = BLOCK;
    __shared__ float local[threads];
    int id = threadIdx.x;
    local[id] = 0;
    int filter = blockIdx.x;
    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < num; i += threads){
            int index = j*num*filters + filter*num + i + id;
            local[id] += (i+id < num) ? x[index] : 0;
        }
    }
    __syncthreads();
    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= num * batch;
    }
}

__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int num, float *variance)
{
    const int threads = BLOCK;
    __shared__ float local[threads];
    int id = threadIdx.x;
    local[id] = 0;
    int filter = blockIdx.x;
    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < num; i += threads){
            int index = j*num*filters + filter*num + i + id;
            local[id] += (i+id < num) ? powf((x[index] - mean[filter]), 2) : 0;
        }
    }
    __syncthreads();
    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= num * batch;
    }
}

void normalize_mean_gpu(float *data, int num, int features, int subdivision, float *mean)
{
    fast_mean_kernel<<<features, BLOCK>>>(data, subdivision, features, num, mean);
}

void normalize_variance_gpu(float *data, int num, int features, int subdivision, float *mean, float *variance)
{
    fast_variance_kernel<<<features, BLOCK>>>(data, mean, subdivision, features, num, variance);
}

__global__ void normalize_kernel(float *data, float *mean, float *variance, int num, int features, int subdivision, float *space)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= subdivision*num*features) return;
    int f = (index/num)%features;
    space[index] = (data[index] - mean[f])/(sqrtf(variance[f] + .00001f));
}

void normalize_gpu(float *data, float *mean, float *variance, int num, int features, int subdivision, float *space)
{
    size_t N = subdivision*num*features;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(data, mean, variance, num, features, subdivision, space);
}

__global__ void fast_mean_delta_kernel(float *n_delta, float *variance, int num, int features, int subdivision, float *mean_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < subdivision; ++j){
        for(i = 0; i < num; i += threads){
            int index = j*num*features + filter*num + i + id;
            local[id] += (i+id < num) ? n_delta[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f/sqrtf(variance[filter] + .00001f));
    }
}

__global__ void  fast_variance_delta_kernel(float *n_delta, float *input, float *mean, float *variance, int num, int features, int subdivision, float *variance_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < subdivision; ++j){
        for(i = 0; i < num; i += threads){
            int index = j*num*features + filter*num + i + id;

            local[id] += (i+id < num) ? n_delta[index]*(input[index] - mean[filter]) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -.5f * powf(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}

void gradient_normalize_mean_gpu(float *n_delta, float *variance, int num, int features, int subdivision, float *mean_delta)
{
    fast_mean_delta_kernel<<<features, BLOCK>>>(n_delta, variance, num, features, subdivision, mean_delta);
}

void gradient_normalize_variance_gpu(float *n_delta, float *input, float *mean, float *variance, int num, int features, int subdivision, float *variance_delta)
{
    fast_variance_delta_kernel<<<features, BLOCK>>>(n_delta, input, mean, variance, num, features, subdivision, variance_delta);
}

__global__ void normalize_delta_kernel(int N, float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int num, int features, int subdivision, float *n_delta, float *l_delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/num)%features;
    l_delta[index] = n_delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (input[index] - mean[f]) / (subdivision*num) + mean_delta[f]/(subdivision*num);
}

void gradient_normalize_gpu(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int num, int features, int subdivision, float *n_delta, float *l_delta)
{
    size_t N = subdivision*features*num;
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, input, mean, variance, mean_delta, variance_delta, num, features, subdivision, n_delta, l_delta);
}

__global__ void gradient_scale_kernel(float *norm_x, float *n_delta, int num, int features, int subdivision, float *space)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < subdivision; ++b){
        for(i = 0; i < num; i += BLOCK){
            int index = p + i + num*(filter + features*b);
            sum += (p+i < num) ? n_delta[index]*norm_x[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) space[filter] += part[i];
    }
}

void gradient_scale_gpu(float *norm_x, float *n_delta, int num, int features, int subdivision, float *space)
{
    gradient_scale_kernel<<<features, BLOCK>>>(norm_x, n_delta, num, features, subdivision, space);
}
