#include "nll_layer_gpu.h"

__global__ void nll_kernel(float *input, float *output, float *truth, int group)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= group) return;
    if (truth[index] == 1) output[0] = -log(input[index]);
    else return;
}

void nll_gpu(float *input, float *output, float *truth, int group)
{
    nll_kernel<<<(group+BLOCK-1)/BLOCK, BLOCK>>>(input, output, truth, group);
}

__global__ void nll_gradient_kernel(float *input, float *truth, float *delta, int group)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= group) return;
    if (truth[index] == 1) delta[index] = -1 / input[index];
    else delta[index] = 0;
}

void nll_gradient(float *input, float *truth, float *delta, int group)
{
    nll_gradient_kernel<<<(group+BLOCK-1)/BLOCK, BLOCK>>>(input, truth, delta, group);
}

void init_nll_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = 1;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->inputs;

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "Mse             Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_nll_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *output = l.output+offset_o;
        float *truth = l.truth+offset_t;
        nll_gpu(input, output, truth, l.group);
    }
    sum_gpu(l.output, l.outputs*num, l.loss);
    multy_gpu(l.loss, 1, (float)1/num, 1);
}

void backward_nll_layer_gpu(Layer l, int num, float *n_delta)
{
    fill_gpu(l.delta, num*l.inputs, 0, 1);
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *delta_l = l.delta+offset_i;
        float *truth = l.truth+offset_t;
        nll_gradient(input, truth, delta_l, l.group);
    }
}

void free_nll_layer_gpu(Layer l)
{
    cudaFree(l.output);
    cudaFree(l.delta);
}
