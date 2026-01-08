#include "global_maxpool_layer_gpu.h"

void init_global_maxpool_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = 1;
    l->output_w = l->input_c;
    l->output_c = 1;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = 0;
    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    cudaMalloc((void**)&l->maxpool_index, subdivision*l->outputs*sizeof(float));

    fprintf(stderr, "Global Max Pooling     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_global_maxpool_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        int *index = l.maxpool_index + offset_o;
        global_maxpool_gpu(input, l.input_h, l.input_w, l.input_c, output, index);
    }
}

void backward_global_maxpool_layer_gpu(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        int *index = l.maxpool_index + offset_o;
        global_maxpool_gradient_gpu(delta_l, l.input_h, l.input_w, l.input_c, delta_n, index);
    }
}

void free_global_maxpool_layer_gpu(Layer l)
{
    cudaFree(l.output);
    cudaFree(l.delta);
    cudaFree(l.maxpool_index);
}