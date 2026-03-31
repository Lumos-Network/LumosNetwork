#include "im2col.h"

void im2col(float *img, int height, int width, int channel, int ksize, int stride, int pad, float *space)
{
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channel * ksize * ksize; //channels_col总行数  height_col*width_col总列数
    for (int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_offset = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h)
        {
            for (int w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride - pad;
                int im_col = w_offset + w * stride - pad;
                int col_index = (height_col * width_col) * c + h * width_col + w;
                if (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width){
                    space[col_index] = 0;
                    continue;
                }
                space[col_index] = img[im_col + width * (im_row + height * c_offset)];
            }
        }
    }
}

void col2im(float *img, int ksize, int stride, int pad, int out_h, int out_w, int out_c, float *space)
{
    int height_col = (out_h + 2 * pad - ksize) / stride + 1;
    int width_col = (out_w + 2 * pad - ksize) / stride + 1;
    for (int i = 0; i < ksize*ksize*out_c; ++i){
        for (int j = 0; j < height_col*width_col; ++j){
            int kernel_h = j/width_col*stride;
            int kernel_w = j%width_col*stride;
            int channel = i/(ksize*ksize);
            int index_h = (i-channel*ksize*ksize)/ksize + kernel_h - pad;
            int index_w = (i-channel*ksize*ksize)%ksize + kernel_w - pad;
            if (index_h >= 0 && index_w >= 0 && index_h < out_h && index_w < out_w){
                space[channel*out_h*out_w+index_h*out_w+index_w] += img[i*height_col*width_col+j];
            }
        }
    }
}
