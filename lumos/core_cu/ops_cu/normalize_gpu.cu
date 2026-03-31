#include "normalize_gpu.h"

__global__ void normalize_mean_kernel(float *data, int num, int features, int subdivision, float *mean)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= features) return;
    mean[index] = 0;
    for (int j = 0; j < subdivision; ++j){
        for (int k = 0; k < num; ++k){
            mean[index] += data[j*num*features+index*num+k];
        }
    }
    mean[index] /= subdivision*num;
}

void normalize_mean_gpu(float *data, int num, int features, int subdivision, float *mean)
{
    normalize_mean_kernel<<<(features+BLOCK-1)/BLOCK, BLOCK>>>(data, num, features, subdivision, mean);
}

__global__ void normalize_variance_kernel(float *data, int num, int features, int subdivision, float *mean, float *variance)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= features) return;
    variance[index] = 0;
    for (int j = 0; j < subdivision; ++j){
        for (int k = 0; k < num; ++k){
            variance[index] += pow(data[j*num*features+index*num+k]-mean[index], 2);
        }
    }
    variance[index] /= subdivision*num;
}

void normalize_variance_gpu(float *data, int num, int features, int subdivision, float *mean, float *variance)
{
    normalize_variance_kernel<<<(features+BLOCK-1)/BLOCK, BLOCK>>>(data, num, features, subdivision, mean, variance);
}

__global__ void normalize_kernel(float *data, float *mean, float *variance, int num, int features, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num*features) return;
    int offset = num;
    int i = index / num;
    int j = index % num;
    float *data_c = data + i*offset;
    float *space_c = space + i*offset;
    space_c[j] = (data_c[j] - mean[i]) / (sqrt(variance[i] + .00001f));
}

void normalize_gpu(float *data, float *mean, float *variance, int num, int features, float *space)
{
    normalize_kernel<<<(num*features+BLOCK-1)/BLOCK, BLOCK>>>(data, mean, variance, num, features, space);
}

__global__ void gradient_normalize_mean_kernel(float *n_delta, float *variance, int num, int features, float *mean_delta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= features) return;
    mean_delta[index] = 0;
    for (int j = 0; j < num; ++j){
        mean_delta[index] += n_delta[index*num+j];
    }
    mean_delta[index] *= (-1./sqrt(variance[index] + .00001f));
}

void gradient_normalize_mean_gpu(float *n_delta, float *variance, int num, int features, float *mean_delta)
{
    gradient_normalize_mean_kernel<<<(features+BLOCK-1)/BLOCK, BLOCK>>>(n_delta, variance, num, features, mean_delta);
}

__global__ void gradient_normalize_variance_kernel(float *n_delta, float *input, float *mean, float *variance, int num, int features, float *variance_delta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= features) return;
    variance_delta[index] = 0;
    for (int j = 0; j < num; ++j){
        variance_delta[index] += n_delta[index*num+j]*(input[index*num+j]-mean[index]);
    }
    variance_delta[index] *= -.5 * pow(variance[index] + .00001f, (float)(-3./2.));
}

void gradient_normalize_variance_gpu(float *n_delta, float *input, float *mean, float *variance, int num, int features, float *variance_delta)
{
    gradient_normalize_variance_kernel<<<(features+BLOCK-1)/BLOCK, BLOCK>>>(n_delta, input, mean, variance, num, features, variance_delta);
}

__global__ void gradient_normalize_kernel(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int num, int features, float *n_delta, float *l_delta)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= features) return;
    for (int j = 0; j < num; ++j){
        l_delta[index*num+j] = n_delta[index*num+j] * 1./(sqrt(variance[index] + .00001f)) + variance_delta[index] * 2. * (input[index*num+j] - mean[index]) / (num) + mean_delta[index]/(num);
    }
}

void gradient_normalize_gpu(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int num, int features, float *n_delta, float *l_delta)
{
    gradient_normalize_kernel<<<(features+BLOCK-1)/BLOCK, BLOCK>>>(input, mean, variance, mean_delta, variance_delta, num, features, n_delta, l_delta);
}

__global__ void gradient_scale_kernel(float *norm_x, float *mean, float *variance, float *delta, int num, int features, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= features) return;
    float sum = 0;
    for (int j = 0; j < num; ++j){
        sum += norm_x[j] * delta[index*num+j];
    }
    space[index] = sum;
}

__global__ void gradient_bias_kernel(float *delta, int num, int features, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= features) return;
    float sum = 0;
    for (int j = 0; j < num; ++j){
        sum += delta[index*num+j];
    }
    space[index] = sum;
}

void gradient_scale_gpu(float *norm_x, float *mean, float *variance, float *delta, int num, int features, float *space)
{
    gradient_scale_kernel<<<(features+BLOCK-1)/BLOCK, BLOCK>>>(norm_x, mean, variance, delta, num, features, space);
}

void gradient_bias_gpu(float *delta, int num, int features, float *space)
{
    gradient_bias_kernel<<<(features+BLOCK-1)/BLOCK, BLOCK>>>(delta, num, features, space);
}
