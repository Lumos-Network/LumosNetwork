#include "inception_layer.h"

Layer *make_inception_layer(Layer **inception, int num, int dim)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = INCEPTION;
    l->inception = inception;
    l->filters = num;
    l->dim = dim;

    l->initialize = init_inception_layer;
    l->forward = forward_inception_layer;
    l->backward = backward_inception_layer;
    l->initializegpu = init_inception_layer_gpu;
    l->forwardgpu = forward_inception_layer_gpu;
    l->backwardgpu = backward_inception_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->sgdoptimizer = NULL;
    l->sgdoptimizergpu = NULL;

    l->refresh = NULL;
    l->refreshgpu = NULL;

    l->saveweights = NULL;
    l->saveweightsgpu = NULL;

    l->zerogradlayer = zerograd_inception_layer;
    l->zerogradlayergpu = zerograd_inception_layer_gpu;

    fprintf(stderr, "Inception       Layer\n");
    return l;
}

void init_inception_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = 0;
    l->output_w = 0;
    l->output_c = 0;
    for (int i = 0; i < l->filters; ++i){
        Layer *inception = l->inception[i];
        if (l->dim == 0){
            l->output_w += inception->input_w;
            l->output_h = inception->input_h;
            l->output_c = inception->input_c;
        } else if (l->dim == 1){
            l->output_w = inception->input_w;
            l->output_h += inception->input_h;
            l->output_c = inception->input_c;
        } else if (l->dim == 2){
            l->output_w = inception->input_w;
            l->output_h = inception->input_h;
            l->output_c += inception->input_c;
        } else {
            fprintf(stderr, "Inception Layer init error! Wrong Dim Set!\n");
            break;
        }
    }
    l->outputs = l->output_w*l->output_h*l->output_c;
    l->workspace = 0;
    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));
    l->inception_input = calloc(l->filters, sizeof(float*));
    l->inception_delta = calloc(l->filters, sizeof(float*));
    l->shapes = calloc(l->filters, sizeof(int*));

    for (int i = 0; i < l->filters; ++i){
        Layer *layer = l->inception[i];
        l->inception_input[i] = layer->input;
        l->inception_delta[i] = layer->delta;
        int *shape = calloc(4, sizeof(int));
        shape[0] = layer->input_w;
        shape[1] = layer->input_h;
        shape[2] = layer->input_c;
        shape[3] = subdivision;
        l->shapes[i] = shape;
    }

    fprintf(stderr, "Inception       Layer    %3d*%3d*%3d\n", l->output_w, l->output_h, l->output_c);
}

void forward_inception_layer(Layer l, int num)
{
    array_cat(l.inception_input, l.shapes, l.filters, 4, l.dim, l.output);
}

void backward_inception_layer(Layer l, int num, float *n_delta)
{
    array_split(n_delta, l.shapes, l.filters, 4, l.dim, l.inception_delta);
}

void zerograd_inception_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
}
