#include "crossentropy_layer.h"

void crossentropy(float *data, float *truth, int num, float *space)
{
    float M = 0;
    float res = 0;
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        res += exp(data[i]-M);
    }
    for (int i = 0; i < num; ++i){
        if (truth[i] == 1){
            space[0] = -data[i]+M+log(res);
            return;
        }
    }
}

void crossentropy_gradient(float *data, float *truth, int num, float *space)
{
    float M = 0;
    float res = 0;
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        space[i] = exp(data[i]-M);
        res += space[i];
    }
    for (int i = 0; i < num; ++i){
        space[i] = space[i]/res-truth[i];
    }
}

Layer *make_crossentropy_layer(int group)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CROSSENTROPY;
    l->group = group;

    l->initialize = init_crossentropy_layer;
    l->forward = forward_crossentropy_layer;
    l->backward = backward_crossentropy_layer;

    l->initializegpu = init_crossentropy_layer_gpu;
    l->forwardgpu = forward_crossentropy_layer_gpu;
    l->backwardgpu = backward_crossentropy_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->update = NULL;
    l->updategpu = NULL;

    l->sgdoptimizer = NULL;
    l->sgdoptimizergpu = NULL;

    l->refresh = NULL;
    l->refreshgpu = NULL;

    l->saveweights = NULL;
    l->saveweightsgpu = NULL;

    l->freelayer = free_crossentropy_layer;
    l->freelayergpu = free_crossentropy_layer_gpu;

    fprintf(stderr, "CrossEntropy             Layer    :    [output=%4d]\n", 1);
    return l;
}

void init_crossentropy_layer(Layer *l, int w, int h, int c, int subdivision)
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

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "CrossEntropy             Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_crossentropy_layer(Layer l, int num)
{
    if (!l.status){
        crossentropy(l.input, l.truth, l.inputs, l.output);
        softmax(l.input, l.group, l.detect);
        return;
    }
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *output = l.output+offset_o;
        float *truth = l.truth+offset_t;
        crossentropy(input, truth, l.inputs, output);
    }
    sum_cpu(l.output, l.outputs*num, l.loss);
    multy_cpu(l.loss, 1, (float)1/num, 1);
}

void backward_crossentropy_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *delta_l = l.delta+offset_i;
        float *truth = l.truth+offset_t;
        crossentropy_gradient(input, truth, l.inputs, delta_l);
    }
}

void free_crossentropy_layer(Layer l)
{
    free(l.output);
    free(l.delta);
}
