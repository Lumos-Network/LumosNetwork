#include "crossentropy_layer_gpu.h"

__global__ void crossentropy_kernel(float *data, int *truth, int w, int h, int c, float *scale, int ignore, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_pixel = w*h;
    float scale_x = 1;
    if (index >= num_pixel) return;
    int t = truth[index];
    if (ignore != -1){
        if (t == ignore){
            space[index] = 0.0;
            return;
        }
    }
    if (scale != NULL){
        scale_x = scale[t];
    }
    float *input = data + index*c;
    float max_val = -INFINITY;
    float sum_exp = 0;
    for (int i = 0; i < c; ++i){
        max_val = max(max_val, input[i]);
    }
    for (int i = 0; i < c; ++i){
        sum_exp += expf(input[i]-max_val);
    }
    space[index] = (-input[t]+max_val+log(sum_exp))*scale_x;
}

void crossentropy_gpu(float *data, int *truth, int w, int h, int c, float *scale, int ignore, float *space)
{
    crossentropy_kernel<<<(w*h+BLOCK-1)/BLOCK, BLOCK>>>(data, truth, w, h, c, scale, ignore, space);
}

__global__ void crossentropy_gradient_kernel(float *data, int *truth, int w, int h, int c, float *scale, int ignore, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_pixel = w*h;
    float scale_x = 1;
    if (index >= num_pixel) return;
    int t = truth[index];
    if (ignore != -1){
        if (t == ignore){
            for (int i = 0; i < c; ++i){
                space[i] = 0.0;
            }
            return;
        }
    }
    if (scale != NULL){
        scale_x = scale[t];
    }
    float *input = data + index*c;
    float max_val = -INFINITY;
    float sum_exp = 0;
    for (int i = 0; i < c; ++i){
        max_val = max(max_val, input[i]);
    }
    for (int i = 0; i < c; ++i){
        space[i] = expf(input[i]-max_val);
        sum_exp += space[i];
    }
    for (int i = 0; i < c; ++i){
        if (i == t) space[i] = (space[i]/sum_exp-1)*scale_x;
        else space[i] = space[i]/sum_exp*scale_x;
    }
}

void crossentropy_gradient_gpu(float *data, int *truth, int w, int h, int c, float *scale, int ignore, float *space)
{
    crossentropy_gradient_kernel<<<(w*h+BLOCK-1)/BLOCK, BLOCK>>>(data, truth, w, h, c, scale, ignore, space);
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
        crossentropy_gpu(l.input, l.truth, l.input_w, l.input_h, l.input_c, l.scale, l.ignore, l.output);
        softmax_gpu(l.input, l.inputs, l.detect);
    } else {
        for (int i = 0; i < num; ++i){
            float *input = l.input + i*l.inputs;
            float *output = l.output + i*l.outputs;
            int *truth = l.truth + i*l.outputs;
            crossentropy_gpu(input, truth, l.input_w, l.input_h, l.input_c, l.scale, l.ignore, output);
        }
    }
    sum_gpu(l.output, l.outputs*num, l.loss);
    multy_gpu(l.loss, 1, (float)1/(l.outputs*num), 1);
}

void backward_crossentropy_layer_gpu(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        float *input = l.input + i*l.inputs;
        float *delta_l = l.delta + i*l.inputs;
        int *truth = l.truth + i*l.outputs;
        crossentropy_gradient_gpu(input, truth, l.input_w, l.input_h, l.input_c, l.scale, l.ignore, delta_l);
    }
}

void zerograd_crossentropy_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
}
