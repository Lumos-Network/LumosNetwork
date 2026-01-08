#include "global_maxpool_layer.h"

Layer *make_global_maxpool_layer()
{
    Layer *l = malloc(sizeof(Layer));
    l->type = GLOBALMAX;

    l->initialize = init_global_maxpool_layer;
    l->forward = forward_global_maxpool_layer;
    l->backward = backward_global_maxpool_layer;

    l->initializegpu = init_global_maxpool_layer_gpu;
    l->forwardgpu = forward_global_maxpool_layer_gpu;
    l->backwardgpu = backward_global_maxpool_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->update = NULL;
    l->updategpu = NULL;

    l->refresh = NULL;
    l->refreshgpu = NULL;

    l->saveweights = NULL;
    l->saveweightsgpu = NULL;

    l->freelayer = free_global_maxpool_layer;
    l->freelayergpu = free_global_maxpool_layer_gpu;

    fprintf(stderr, "Global Max Pooling     Layer\n");
    return l;
}

void init_global_maxpool_layer(Layer *l, int w, int h, int c, int subdivision)
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
    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    l->maxpool_index = calloc(subdivision*l->outputs, sizeof(float));

    fprintf(stderr, "Global Max Pooling     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_global_maxpool_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        int *index = l.maxpool_index + offset_o;
        global_maxpool(input, l.input_h, l.input_w, l.input_c, output, index);
    }
}

void backward_global_maxpool_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        int *index = l.maxpool_index + offset_o;
        global_maxpool_gradient(delta_l, l.input_h, l.input_w, l.input_c, delta_n, index);
    }
}

void free_global_maxpool_layer(Layer l)
{
    free(l.output);
    free(l.delta);
    free(l.maxpool_index);
}
