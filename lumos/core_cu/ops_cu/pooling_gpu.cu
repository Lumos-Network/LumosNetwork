#include "pooling_gpu.h"

__global__ void avgpool_kernel(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    int k = index / (out_h*out_w);
    int i = (index % (out_h*out_w)) / out_w;
    int j = (index % (out_h*out_w)) % out_w;
    if (k >= c || i >= out_h || j >= out_w) return;
    int x = i*stride;
    int y = j*stride;
    float temp = 0;
    for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
        for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
            int index_i = x + ksize_i - pad;
            int index_j = y + ksize_j - pad;
            if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
            temp += im[k*h*w + index_i*w + index_j];
        }
    }
    space[k*out_h*out_w + i*out_w + j] = temp / (ksize*ksize);
}

void avgpool_gpu(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    avgpool_kernel<<<(out_h*out_w*c + BLOCK - 1)/BLOCK, BLOCK>>>(im, h, w, c, ksize, stride, pad, space);
}

__global__ void maxpool_kernel(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index)
{
    int index_g = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    int k = index_g / (out_h*out_w);
    int i = (index_g % (out_h*out_w)) / out_w;
    int j = (index_g % (out_h*out_w)) % out_w;
    if (k >= c || i >= out_h || j >= out_w) return;
    int x = i*stride;
    int y = j*stride;
    int max_index = -1;
    float max = -FLT_MAX;
    for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
        for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
            int index_i = x + ksize_i - pad;
            int index_j = y + ksize_j - pad;
            if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
            if (im[k*h*w + index_i*w + index_j] > max){
                max = im[k*h*w + index_i*w + index_j];
                max_index = k*h*w + index_i*w + index_j;
            }
        }
    }
    space[k*out_h*out_w + i*out_w + j] = max;
    index[k*out_h*out_w + i*out_w + j] = max_index;
}

void maxpool_gpu(float *im, int h, int w, int c, int ksize, int stride, int pad, float *space, int *index)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    maxpool_kernel<<<(out_h*out_w*c + BLOCK - 1)/BLOCK, BLOCK>>>(im, h, w, c, ksize, stride, pad, space, index);
}

__global__ void avgpool_gradient_kernel(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    int k = index / (out_h*out_w);
    int i = (index % (out_h*out_w)) / out_w;
    int j = (index % (out_h*out_w)) % out_w;
    if (k >= c || i >= out_h || j >= out_w) return;
    int x = i*stride;
    int y = j*stride;
    for (int ksize_i = 0; ksize_i < ksize; ++ksize_i){
        for (int ksize_j = 0; ksize_j < ksize; ++ksize_j){
            int index_i = x + ksize_i - pad;
            int index_j = y + ksize_j - pad;
            if (index_i <= -1 || index_i >= h || index_j <= -1 || index_j >= w) continue;
            delta_l[k*h*w + index_i*w + index_j] += delta_n[k*out_h*out_w + i*out_w + j] / (ksize*ksize);
        }
    }
}

void avgpool_gradient_gpu(float *delta_l, int h, int w, int c, int ksize, int stride, int pad, float *delta_n)
{
    int out_h = (h + 2 * pad - ksize) / stride + 1;
    int out_w = (w + 2 * pad - ksize) / stride + 1;
    avgpool_gradient_kernel<<<(out_h*out_w*c + BLOCK - 1)/BLOCK, BLOCK>>>(delta_l, h, w, c, ksize, stride, pad, delta_n);
}

__global__ void maxpool_gradient_kernel(float *delta_l, int h, int w, int c, float *delta_n, int *index)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0) return;
    for (int j = 0; j < h*w*c; ++j){
        delta_l[index[j]] += delta_n[j];
    }
}

void maxpool_gradient_gpu(float *delta_l, int h, int w, int c, float *delta_n, int *index)
{
    maxpool_gradient_kernel<<<(h*w*c + BLOCK - 1)/BLOCK, BLOCK>>>(delta_l, h, w, c, delta_n, index);
}

__global__ void global_avgpool_kernel(float *im, int h, int w, int c, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int k = index / (h*w);
    int i = (index % (h*w)) / w;
    int j = (index % (h*w)) % w;
    if (k >= c || i >= h || j >= w) return;
    space[k] += im[k*h*w + i*w + j] / (h*w);
}
