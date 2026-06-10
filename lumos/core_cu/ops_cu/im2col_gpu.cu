#include "im2col_gpu.h"

__global__ void im2col_kernel(float *img, int height, int width, int channel, int ksize, int stride, int pad, int dilation, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int dksize = ksize*(dilation+1)-dilation;
    int height_col = (height + 2 * pad - dksize) / stride + 1;
    int width_col = (width + 2 * pad - dksize) / stride + 1;
    int channels_col = channel * ksize * ksize;
    if (index >= height_col*width_col*channels_col) return;
    int c = index / (height_col*width_col);
    int h = (index % (height_col*width_col)) / width_col;
    int w = (index % (height_col*width_col)) % width_col;
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_offset = c / ksize / ksize;
    int im_row = h_offset + h * stride + h_offset * dilation;
    int im_col = w_offset + w * stride + w_offset * dilation;
    int col_index = (height_col * width_col) * c + h * width_col + w;
    if (im_row-pad < 0 || im_col-pad < 0 || im_row-pad >= height || im_col-pad >= width){
        space[col_index] = 0;
        return;
    }
    space[col_index] = img[im_col + width * (im_row + height * c_offset - pad) - pad];
}

void im2col_gpu(float *img, int height, int width, int channel, int ksize, int stride, int pad, int dilation, float *space)
{
    int dksize = ksize*(dilation+1)-dilation;
    int height_col = (height + 2 * pad - dksize) / stride + 1;
    int width_col = (width + 2 * pad - dksize) / stride + 1;
    int channels_col = channel * ksize * ksize;
    im2col_kernel<<<(height_col*width_col*channels_col+BLOCK-1)/BLOCK, BLOCK>>>(img, height, width, channel, ksize, stride, pad, dilation, space);
}

__global__ void col2im_kernel(int n, float *img, int ksize, int stride, int pad, int dilation, int out_h, int out_w, int out_c, int height_col, int width_col, float *space)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        float val = 0;
        int w = index % out_w + pad;
        int h = (index / out_w) % out_h + pad;
        int c = index / (out_w * out_h);
        int kernel_extent = (ksize - 1) * dilation + 1;
        // compute the start and end of the output
        int w_col_start = (w < kernel_extent) ? 0 : (w - kernel_extent) / stride + 1;
        int w_col_end = min(w / stride + 1, width_col);
        int h_col_start = (h < kernel_extent) ? 0 : (h - kernel_extent) / stride + 1;
        int h_col_end = min(h / stride + 1, height_col);
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                int h_k = (h - h_col * stride);
                int w_k = (w - w_col * stride);
                if (h_k % dilation == 0 && w_k % dilation == 0) {
                    h_k /= dilation;
                    w_k /= dilation;
                    int data_col_index = (((c * ksize + h_k) * ksize + w_k) *height_col + h_col) * width_col + w_col;
                    val += img[data_col_index];
                }
            }
        }
        space[index] += val;
    }
}

void col2im_gpu(float *img, int ksize, int stride, int pad, int dilation, int out_h, int out_w, int out_c, float *space)
{
    int dksize = ksize*(dilation+1)-dilation;
    int height_col = (out_h + 2 * pad - dksize) / stride + 1;
    int width_col = (out_w + 2 * pad - dksize) / stride + 1;
    int num_kernels = out_h*out_w*out_c;
    col2im_kernel<<<(ksize*ksize*out_c*height_col*width_col+BLOCK-1)/BLOCK, BLOCK>>>(num_kernels, img, ksize, stride, pad, dilation+1, out_h, out_w, out_c, height_col, width_col, space);
}
