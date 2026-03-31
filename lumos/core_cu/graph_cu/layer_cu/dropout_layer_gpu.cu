#include "dropout_layer_gpu.h"

void init_dropout_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->inputs;

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));
    cudaMalloc((void**)&l->dropout_rand, subdivision*l->inputs*sizeof(float));
    float *rand_l = (float*)calloc(l->outputs, sizeof(float));
    for (int i = 0; i < l->outputs; ++i){
        rand_l[i] = rand_uniform(0, 1);
    }
    cudaMemcpy(l->dropout_rand, rand_l, l->outputs*sizeof(float), cudaMemcpyHostToDevice);
    fprintf(stderr, "Dropout         Layer\n");
}

void forward_dropout_layer_gpu(Layer l, int num)
{
    if (!l.status){
        cudaMemcpy(l.output, l.input, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }
    float *dropout_rand = (float*)calloc(num*l.inputs, sizeof(float));
    for (int i = 0; i < num*l.inputs; ++i){
        dropout_rand[i] = rand_uniform(0, 1);
    }
    cudaMemcpy(l.dropout_rand, dropout_rand, num*l.inputs*sizeof(float), cudaMemcpyHostToDevice);
    dropout_gpu(l, num);
}

void backward_dropout_layer_gpu(Layer l, int num, float *n_delta)
{
    dropout_gradient_gpu(l, num, n_delta);
}

__global__ void dropout_kernel(Layer l, int num, float scale)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num*l.inputs) return;
    float r = l.dropout_rand[index];
    if (r < l.probability) l.output[index] = 0;
    else l.output[index] = l.input[index] * scale;
}

__global__ void dropout_gradient_kernel(Layer l, int num, float *n_delta, float scale)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num*l.inputs) return;
    float r = l.dropout_rand[index];
    if (r < l.probability) l.delta[index] = 0;
    else l.delta[index] = n_delta[index] * scale;
}

void dropout_gpu(Layer l, int num)
{
    int size = num * l.inputs;
    float scale = 1. / (1.-l.probability);
    dropout_kernel<<<(size+BLOCK-1)/BLOCK, BLOCK>>>(l, num, scale);
}

void dropout_gradient_gpu(Layer l, int num, float *n_delta)
{
    int size = num * l.inputs;
    float scale = 1. / (1.-l.probability);
    dropout_gradient_kernel<<<(size+BLOCK-1)/BLOCK, BLOCK>>>(l, num, n_delta, scale);
}

void zerograd_dropout_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
}
