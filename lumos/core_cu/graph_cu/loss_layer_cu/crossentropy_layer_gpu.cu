#include "crossentropy_layer_gpu.h"

__global__ void crossentropy_kernel(float *data, float *truth, int num, float *scale, float *max, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    if (truth[index] == 1){
        space[0] = (-data[index]+max[0]+log(space[0]))*scale[0];
    }
}

void crossentropy_gpu(float *data, float *truth, int num, float *scale, int ignore, float *workspace, float *space)
{
    float *max;
    float *scale_x;
    if (ignore != -1){
        if (truth[ignore] == 1){
            fill_gpu(space, 1, 0, 1);
            return;
        }
    }
    cudaMalloc((void**)&max, sizeof(float));
    cudaMalloc((void**)&scale_x, sizeof(float));
    if (scale != NULL){
        gemm_gpu(0, 0, 1, num, num, 1, 1, truth, scale, scale_x);
    } else {
        fill_gpu(scale_x, 1, 1, 1);
    }
    max_gpu(data, num, max);
    exp_list_gpu(data, num, workspace, max);
    sum_gpu(workspace, num, space);
    crossentropy_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(data, truth, num, scale_x, max, space);
    cudaFree(max);
    cudaFree(scale_x);
}

__global__ void crossentropy_gradient_kernel(float *truth, int num, float *scale, float *res, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    space[index] = (space[index]/res[0]-truth[index])*scale[0]; 
}

void crossentropy_gradient_gpu(float *data, float *truth, int num, float *scale, int ignore, float *workspace, float *space)
{
    float *max;
    float *scale_x;
    if (ignore != -1){
        if (truth[ignore] == 1){
            fill_gpu(space, num, 0, 1);
            return;
        }
    }
    cudaMalloc((void**)&max, sizeof(float));
    cudaMalloc((void**)&scale_x, sizeof(float));
    if (scale != NULL){
        gemm_gpu(0, 0, 1, num, num, 1, 1, truth, scale, scale_x);
    } else {
        fill_gpu(scale_x, 1, 1, 1);
    }
    max_gpu(data, num, max);
    exp_list_gpu(data, num, space, max);
    sum_gpu(space, num, workspace);
    crossentropy_gradient_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(truth, num, scale_x, workspace, space);
    cudaFree(max);
    cudaFree(scale_x);
}

void init_crossentropy_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = l->input_h;
    l->output_w = l->input_w;
    l->output_c = 1;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->inputs;

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "CrossEntropy    Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_crossentropy_layer_gpu(Layer l, int num)
{
    if (!l.status){
        for (int i = 0; i < l.input_h*l.input_w; ++i){
            float *input = l.input+i*l.input_c;
            float *detect = l.detect+i*l.input_c;
            float *output = l.output+i;
            crossentropy_gpu(input, l.truth, l.input_c, l.scale, l.ignore, l.workspace, output);
            softmax_exp_sum_gpu(input, l.input_c, l.workspace, l.workspace+l.input_c);
            softmax_gpu(l.workspace, l.input_c, detect, l.workspace+l.input_c);
        }
        return;
    }
    for (int i = 0; i < num; ++i){
        for (int j = 0; j < l.input_h*l.input_w; ++j){
            float *input = l.input+(i*l.input_h*l.input_w+j)*l.input_c;
            float *truth = l.truth+i*l.input_c;
            float *output = l.output+i*l.output_h*l.output_w+j;
            crossentropy_gpu(input, truth, l.input_c, l.scale, l.ignore, l.workspace, output);
        }
    }
    sum_gpu(l.output, l.outputs*num, l.loss);
    multy_gpu(l.loss, 1, (float)1/num, 1);
}

void backward_crossentropy_layer_gpu(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        for (int j = 0; j < l.input_h*l.input_w; ++j){
            float *input = l.input+(i*l.input_h*l.input_w+j)*l.input_c;
            float *truth = l.truth+i*l.input_c;
            float *delta_l = l.delta+(i*l.input_h*l.input_w+j)*l.input_c;
            crossentropy_gradient_gpu(input, truth, l.input_c, l.scale, l.ignore, l.workspace, delta_l);
        }
    }
}

void zerograd_crossentropy_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
}
