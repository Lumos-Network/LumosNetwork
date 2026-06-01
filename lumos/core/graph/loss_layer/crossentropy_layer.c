#include "crossentropy_layer.h"

void max_channel_cpu(float *data, int w, int h, int c, int index, float *space)
{
    float max_val = -INFINITY;
    for (int i = 0; i < c; ++i){
        if (data[i*w*h+index] >= max_val) max_val = data[i*w*h+index];
    }
    space[0] = max_val;
}

void softmax_channel_cpu(float *data, int w, int h, int c, int index, float *space)
{
    float M = 0;
    float res = 0;
    max_channel_cpu(data, w, h, c, index, &M);
    for (int i = 0; i < c; ++i){
        res += expf(data[i*w*h+index]-M);
    }
    for (int i = 0; i < c; ++i){
        space[i*w*h+index] = expf(data[i*w*h+index]-M)/res;
    }
}

void crossentropy(float *data, int *truth, int w, int h, int c, int index, float *scale, int ignore, float *space)
{
    float M = 0;
    float res = 0;
    float scale_x = 1;
    int target = truth[index];
    if (ignore != -1){
        if (target == ignore){
            space[index] = 0.0;
            return;
        }
    }
    // if (scale != NULL){
    //     scale_x = scale[target];
    // }
    max_channel_cpu(data, w, h, c, index, &M);
    for (int i = 0; i < c; ++i){
        res += expf(data[i*w*h+index]-M);
    }
    space[index] = (-data[target*w*h+index]+M+log(res))*scale_x;
}

void crossentropy_gradient(float *data, int *truth, int w, int h, int c, int index, float *scale, int ignore, float *space)
{
    float M = 0;
    float res = 0;
    float scale_x = 1;
    int target = truth[index];
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
    max_channel_cpu(data, w, h, c, index, &M);
    for (int i = 0; i < c; ++i){
        space[i*w*h+index] = expf(data[i*w*h+index]-M);
        res += space[i*w*h+index];
    }
    for (int i = 0; i < c; ++i){
        if (i == target){
            space[i*w*h+index] = (space[i*w*h+index]/res-1)*scale_x/(w*h);
        } else {
            space[i*w*h+index] = space[i*w*h+index]/res*scale_x/(w*h);
        }
    }
}

Layer *make_crossentropy_layer(float *scale, int ignore)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CROSSENTROPY;
    l->scale = scale;
    l->ignore = ignore;

    l->initialize = init_crossentropy_layer;
    l->forward = forward_crossentropy_layer;
    l->backward = backward_crossentropy_layer;

    l->initializegpu = init_crossentropy_layer_gpu;
    l->forwardgpu = forward_crossentropy_layer_gpu;
    l->backwardgpu = backward_crossentropy_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->sgdoptimizer = NULL;
    l->sgdoptimizergpu = NULL;

    l->refresh = NULL;
    l->refreshgpu = NULL;

    l->saveweights = NULL;
    l->saveweightsgpu = NULL;

    l->zerogradlayer = zerograd_crossentropy_layer;
    l->zerogradlayergpu = zerograd_crossentropy_layer_gpu;

    fprintf(stderr, "CrossEntropy    Layer    :    [output=%4d]\n", 1);
    return l;
}

void init_crossentropy_layer(Layer *l, int w, int h, int c, int subdivision)
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

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "CrossEntropy    Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_crossentropy_layer(Layer l, int num)
{
    if (!l.status){
        for (int i = 0; i < l.input_h*l.input_w; ++i){
            crossentropy(l.input, l.truth, l.input_w, l.input_h, l.input_c, i, l.scale, l.ignore, l.output);
            softmax_channel_cpu(l.input, l.input_w, l.input_h, l.input_c, i, l.detect);
        }
        return;
    } else {
        for (int i = 0; i < num; ++i){
            for (int j = 0; j < l.input_h*l.input_w; ++j){
                float *input = l.input+i*l.inputs;
                int *truth = l.truth+i*l.truth_num;
                float *output = l.output+i*l.outputs;
                crossentropy(input, truth, l.input_w, l.input_h, l.input_c, j, l.scale, l.ignore, output);
            }
        }
    }
    sum_cpu(l.output, l.outputs*num, l.loss);
    multy_cpu(l.loss, 1, (float)1/(l.outputs*num), 1);
}

void backward_crossentropy_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        for (int j = 0; j < l.input_h*l.input_w; ++j){
            float *input = l.input+i*l.inputs;
            int *truth = l.truth+i*l.truth_num;
            float *delta_l = l.delta+i*l.inputs;
            crossentropy_gradient(input, truth, l.input_w, l.input_h, l.input_c, j, l.scale, l.ignore, delta_l);
        }
    }
}

void zerograd_crossentropy_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
}
