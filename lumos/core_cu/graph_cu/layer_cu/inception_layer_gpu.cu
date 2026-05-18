#include "inception_layer_gpu.h"

void init_inception_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
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
    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));
    l->inception_input = (float**)malloc(l->filters*sizeof(float*));
    l->inception_delta = (float**)malloc(l->filters*sizeof(float*));
    l->shapes = (int**)malloc(l->filters*sizeof(int*));

    for (int i = 0; i < l->filters; ++i){
        Layer *layer = l->inception[i];
        l->inception_input[i] = layer->input;
        l->inception_delta[i] = layer->delta;
        int *shape = (int*)malloc(4*sizeof(int));
        shape[0] = layer->input_w;
        shape[1] = layer->input_h;
        shape[2] = layer->input_c;
        shape[3] = subdivision;
        l->shapes[i] = (int*)shape;
    }

    fprintf(stderr, "Inception       Layer    %3d*%3d*%3d\n", l->output_w, l->output_h, l->output_c);
}

void forward_inception_layer_gpu(Layer l, int num)
{
    array_cat_gpu(l.inception_input, l.shapes, l.filters, 4, l.dim, l.output);
}

void backward_inception_layer_gpu(Layer l, int num, float *n_delta)
{
    array_split_gpu(n_delta, l.shapes, l.filters, 4, l.dim, l.inception_delta);
}

void zerograd_inception_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
}
