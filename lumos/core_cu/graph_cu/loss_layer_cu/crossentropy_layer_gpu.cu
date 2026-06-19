#include "crossentropy_layer_gpu.h"

__global__ void crossentropy_kernel(float *data, float *truth, int w, int h, int c, float *scale, int ignore, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float scale_x = 1;
    if (index >= w*h) return;
    int target = (int)truth[index];
    if (ignore != -1){
        if (target == ignore){
            space[index] = 0.0;
            return;
        }
    }
    if (scale != NULL){
        scale_x = scale[target];
    }
    float max_val = -INFINITY;
    float sum_exp = 0;
    for (int i = 0; i < c; ++i){
        max_val = max(max_val, data[i*w*h+index]);
    }
    for (int i = 0; i < c; ++i){
        sum_exp += expf(data[i*w*h+index]-max_val);
    }
    space[index] = (-data[target*w*h+index]+max_val+log(sum_exp))*scale_x;
}

void crossentropy_gpu(float *data, float *truth, int w, int h, int c, float *scale, int ignore, float *space)
{
    crossentropy_kernel<<<(w*h+BLOCK-1)/BLOCK, BLOCK>>>(data, truth, w, h, c, scale, ignore, space);
}

__global__ void crossentropy_gradient_kernel(float *data, float *truth, int w, int h, int c, float *scale, int ignore, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float scale_x = 1;
    if (index >= w*h) return;
    int target = (int)truth[index];
    if (ignore != -1){
        if (target == ignore){
            for (int i = 0; i < c; ++i){
                space[i*w*h+index] = 0.0;
            }
            return;
        }
    }
    if (scale != NULL){
        scale_x = scale[target];
    }
    float max_val = -INFINITY;
    float sum_exp = 0;
    for (int i = 0; i < c; ++i){
        max_val = max(max_val, data[i*w*h+index]);
    }
    for (int i = 0; i < c; ++i){
        space[i*w*h+index] = expf(data[i*w*h+index]-max_val);
        sum_exp += space[i*w*h+index];
    }
    for (int i = 0; i < c; ++i){
        if (i == target) space[i*w*h+index] = (space[i*w*h+index]/sum_exp-1)/(w*h)*scale_x;
        else space[i*w*h+index] = (space[i*w*h+index]/sum_exp)/(w*h)*scale_x;
    }
}

void crossentropy_gradient_gpu(float *data, float *truth, int w, int h, int c, float *scale, int ignore, float *space)
{
    crossentropy_gradient_kernel<<<(w*h+BLOCK-1)/BLOCK, BLOCK>>>(data, truth, w, h, c, scale, ignore, space);
}

__global__ void softmax_channel_kernel(float *data, int w, int h, int c, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= w*h) return;
    float max_val = -INFINITY;
    float sum_exp = 0;
    for (int i = 0; i < c; ++i){
        max_val = max(max_val, data[i*w*h+index]);
    }
    for (int i = 0; i < c; ++i){
        sum_exp += expf(data[i*w*h+index] - max_val);
    }
    for (int i = 0; i < c; ++i){
        space[i*w*h+index] = expf(data[i*w*h+index] - max_val) / sum_exp;
    }
}

void softmax_channel_gpu(float *data, int w, int h, int c, float *space)
{
    softmax_channel_kernel<<<(w*h+BLOCK-1)/BLOCK, BLOCK>>>(data, w, h, c, space);
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

    l->workspace_size = 0;

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    if (l->scale != NULL){
        float *scale = NULL;
        cudaMalloc((void**)&scale, l->class_num*sizeof(float));
        cudaMemcpy(scale, l->scale, l->class_num*sizeof(float), cudaMemcpyHostToDevice);
        free(l->scale);
        l->scale = scale;
    }

    fprintf(stderr, "CrossEntropy    Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_crossentropy_layer_gpu(Layer l, int num)
{
    if (!l.status){
        crossentropy_gpu(l.input, l.truth, l.input_w, l.input_h, l.input_c, l.scale, l.ignore, l.output);
        cudaMemcpy(l.detect, l.input, l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        for (int i = 0; i < num; ++i){
            float *input = l.input + i*l.inputs;
            float *output = l.output + i*l.outputs;
            float *truth = l.truth + i*l.truth_num;
            crossentropy_gpu(input, truth, l.input_w, l.input_h, l.input_c, l.scale, l.ignore, output);
        }
    }
    sum_gpu(l.output, l.outputs*num, l.loss);
    multy_gpu(l.loss, 1, (float)1/(l.outputs*num), 1);
}

void backward_crossentropy_layer_gpu(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        float *input = l.input+i*l.inputs;
        float *truth = l.truth+i*l.truth_num;
        float *delta_l = l.delta+i*l.inputs;
        crossentropy_gradient_gpu(input, truth, l.input_w, l.input_h, l.input_c, l.scale, l.ignore, delta_l);
    }
}

void zerograd_crossentropy_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
    fill_gpu(l.loss, 1, 0, 1);
}
