#include "im2col_gpu.h"

__global__ void im2col_kernel(float *img, int height, int width, int channel, int ksize, int stride, int pad, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channel * ksize * ksize;
    if (index >= height_col*width_col*channels_col) return;
    int c = index / (height_col*width_col);
    int h = (index % (height_col*width_col)) / width_col;
    int w = (index % (height_col*width_col)) % width_col;
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_offset = c / ksize / ksize;
    int im_row = h_offset + h * stride;
    int im_col = w_offset + w * stride;
    int col_index = (height_col * width_col) * c + h * width_col + w;
    if (im_row-pad < 0 || im_col-pad < 0 || im_row-pad >= height || im_col-pad >= width){
        space[col_index] = 0;
        return;
    }
    space[col_index] = img[im_col + width * (im_row + height * c_offset - pad) - pad];
}

void im2col_gpu(float *img, int height, int width, int channel, int ksize, int stride, int pad, float *space)
{
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channel * ksize * ksize;
    im2col_kernel<<<(height_col*width_col*channels_col+BLOCK-1)/BLOCK, BLOCK>>>(img, height, width, channel, ksize, stride, pad, space);
}

__global__ void col2im_kernel(int n, float *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, int height_col, int width_col, float *space)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        float val = 0;
        int w = index % out_w + pad;
        int h = (index / out_w) % out_h + pad;
        int c = index / (out_w * out_h);
        // compute the start and end of the output
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        // equivalent implementation
        int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        int coeff_w_col = (1 - stride * height_col * width_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += img[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        space[index] += val;
    }
}

void col2im_gpu(float *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, float *space)
{
    int height_col = (out_h + 2 * pad - ksize) / stride + 1;
    int width_col = (out_w + 2 * pad - ksize) / stride + 1;
    int num_kernels = out_h * out_w * out_c;
    col2im_kernel<<<(ksize*ksize*out_c*height_col*width_col+BLOCK-1)/BLOCK, BLOCK>>>(num_kernels, img, ksize, stride, pad, out_h, out_w, out_c, height_col, width_col, space);
}
