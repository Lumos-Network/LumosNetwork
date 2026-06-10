#include "interpolate_layer.h"

void interpolate(float *img, int height, int width, int channel, int row, int col, float *space)
{
    if (height == row && width == col)
    {
        memcpy(space, img, height * width * channel * sizeof(float));
        return;
    }
    float w_scale = (float)(width - 1) / (col - 1);
    float h_scale = (float)(height - 1) / (row - 1);
    for (int i = 0; i < channel; ++i){
        for (int j = 0; j < row; ++j){
            for (int k = 0; k < col; ++k){
                float hx = j*h_scale;
                float wx = k*w_scale;
                int ih = (int)hx;
                int iw = (int)wx;
                float dh = hx - ih;
                float dw = wx - iw;
                float val_0 = (1-dh) * (1-dw) * img[i*height*width + ih*width + iw];
                float val_1 = (1-dh) * dw * img[i*height*width + ih*width + (iw+1 < width ? iw+1 : iw)];
                float val_2 = dh * (1-dw) * img[i*height*width + (ih+1 < height ? ih+1 : ih)*width + iw];
                float val_3 = dh * dw * img[i*height*width + (ih+1 < height ? ih+1 : ih)*width + (iw+1 < width ? iw+1 : iw)];
                space[i*row*col + j*col + k] = val_0 + val_1 + val_2 + val_3;
            }
        }
    }
}

void interpolate_gradient(float *img, int row, int col, int channel, int height, int width, float *space)
{
    if (height == row && width == col)
    {
        memcpy(space, img, height * width * channel * sizeof(float));
        return;
    }
    float w_scale = (float)(width - 1) / (col - 1);
    float h_scale = (float)(height - 1) / (row - 1);
    for (int i = 0; i < channel; ++i){
        for (int j = 0; j < row; ++j){
            for (int k = 0; k < col; ++k){
                float hx = j*h_scale;
                float wx = k*w_scale;
                int ih = (int)hx;
                int iw = (int)wx;
                float dh = hx - ih;
                float dw = wx - iw;
                space[i*height*width + ih*width + iw] += (1-dh) * (1-dw) * img[i*row*col + j*col + k];
                space[i*height*width + ih*width + (iw+1 < width ? iw+1 : iw)] += (1-dh) * dw * img[i*row*col + j*col + k];
                space[i*height*width + (ih+1 < height ? ih+1 : ih)*width + iw] += dh * (1-dw) * img[i*row*col + j*col + k];
                space[i*height*width + (ih+1 < height ? ih+1 : ih)*width + (iw+1 < width ? iw+1 : iw)] += dh * dw * img[i*row*col + j*col + k];
            }
        }
    }
}

Layer *make_interpolate_layer(int height, int width)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = INTERPOLATE;
    l->output_h = height;
    l->output_w = width;

    l->initialize = init_interpolate_layer;
    l->forward = forward_interpolate_layer;
    l->backward = backward_interpolate_layer;

    l->initializegpu = init_interpolate_layer_gpu;
    l->forwardgpu = forward_interpolate_layer_gpu;
    l->backwardgpu = backward_interpolate_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->sgdoptimizer = NULL;
    l->sgdoptimizergpu = NULL;

    l->refresh = NULL;
    l->refreshgpu = NULL;

    l->saveweights = NULL;
    l->saveweightsgpu = NULL;

    l->zerogradlayer = zerograd_interpolate_layer;
    l->zerogradlayergpu = zerograd_interpolate_layer_gpu;

    fprintf(stderr, "Interpolate     Layer\n");
    return l;
}

void init_interpolate_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_c = l->input_c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = 0;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "Interpolate     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_interpolate_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        interpolate(l.input + i*l.inputs, l.input_h, l.input_w, l.input_c, l.output_h, l.output_w, l.output + i*l.outputs);
    }
}

void backward_interpolate_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        interpolate_gradient(n_delta + i*l.outputs, l.output_h, l.output_w, l.output_c, l.input_h, l.input_w, l.delta + i*l.inputs);
    }
}

void zerograd_interpolate_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
}
