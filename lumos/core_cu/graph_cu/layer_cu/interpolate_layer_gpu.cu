#include "interpolate_layer_gpu.h"

__global__ void interpolate_kernel(float *img, int height, int width, int channel, int row, int col, int batch, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= row*col*channel*batch) return;
    int index_im = index / (row*col*channel);
    float *img_ptr = img + index_im*height*width*channel;
    float *space_ptr = space + index_im*row*col*channel;
    index = index % (row*col*channel);
    int i = index / (row*col);
    int j = (index % (row*col)) / col;
    int k = index % col;
    float w_scale = (float)(width - 1) / (col - 1);
    float h_scale = (float)(height - 1) / (row - 1);
    float hx = j*h_scale;
    float wx = k*w_scale;
    int ih = (int)hx;
    int iw = (int)wx;
    float dh = hx - ih;
    float dw = wx - iw;
    float val_0 = (1-dh) * (1-dw) * img_ptr[i*height*width + ih*width + iw];
    float val_1 = (1-dh) * dw * img_ptr[i*height*width + ih*width + (iw+1 < width ? iw+1 : iw)];
    float val_2 = dh * (1-dw) * img_ptr[i*height*width + (ih+1 < height ? ih+1 : ih)*width + iw];
    float val_3 = dh * dw * img_ptr[i*height*width + (ih+1 < height ? ih+1 : ih)*width + (iw+1 < width ? iw+1 : iw)];
    space_ptr[i*row*col + j*col + k] = val_0 + val_1 + val_2 + val_3;
}

__global__ void interpolate_gradient_kernel(float *img, int row, int col, int channel, int height, int width, int batch, float *space)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= row*col*channel*batch) return;
    int index_im = index / (row*col*channel);
    float *img_ptr = img + index_im*row*col*channel;
    float *space_ptr = space + index_im*height*width*channel;
    index = index % (row*col*channel);
    int i = index / (row*col);
    int j = (index % (row*col)) / col;
    int k = index % col;
    float w_scale = (float)(width - 1) / (col - 1);
    float h_scale = (float)(height - 1) / (row - 1);
    float hx = j*h_scale;
    float wx = k*w_scale;
    int ih = (int)hx;
    int iw = (int)wx;
    float dh = hx - ih;
    float dw = wx - iw;
    float val_0 = (1-dh) * (1-dw) * img_ptr[i*row*col + j*col + k];
    float val_1 = (1-dh) * dw * img_ptr[i*row*col + j*col + k];
    float val_2 = dh * (1-dw) * img_ptr[i*row*col + j*col + k];
    float val_3 = dh * dw * img_ptr[i*row*col + j*col + k];
    atomicAdd(space_ptr+i*height*width + ih*width + iw, val_0);
    atomicAdd(space_ptr+i*height*width + ih*width + (iw+1 < width ? iw+1 : iw), val_1);
    atomicAdd(space_ptr+i*height*width + (ih+1 < height ? ih+1 : ih)*width + iw, val_2);
    atomicAdd(space_ptr+i*height*width + (ih+1 < height ? ih+1 : ih)*width + (iw+1 < width ? iw+1 : iw), val_3);
}

void init_interpolate_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_c = l->input_c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = 0;

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "Interpolate     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_interpolate_layer_gpu(Layer l, int num)
{
    if (l.input_w == l.output_w && l.input_h == l.output_h) {
        cudaMemcpy(l.output, l.input, num*l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }
    int batch = num;
    int channel = l.input_c;
    int row = l.output_h;
    int col = l.output_w;
    int size = row*col*channel*batch;
    int block_num = (size + BLOCK-1) / BLOCK;
    interpolate_kernel<<<block_num, BLOCK>>>(l.input, l.input_h, l.input_w, l.input_c, l.output_h, l.output_w, batch, l.output);
}

void backward_interpolate_layer_gpu(Layer l, int num, float *n_delta)
{
    if (l.input_w == l.output_w && l.input_h == l.output_h) {
        cudaMemcpy(l.delta, n_delta, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }
    int batch = num;
    int channel = l.input_c;
    int row = l.output_h;
    int col = l.output_w;
    int size = row*col*channel*batch;
    int block_num = (size + BLOCK-1) / BLOCK;
    interpolate_gradient_kernel<<<block_num, BLOCK>>>(n_delta, l.output_h, l.output_w, l.output_c, l.input_h, l.input_w, batch, l.delta);
}

void zerograd_interpolate_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
}
