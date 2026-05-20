#include "crossentropy_layer.h"

void crossentropy(float *data, int *truth, int num, float *scale, int ignore, float *space)
{
    float M = 0;
    float res = 0;
    float scale_x = 1;
    int index = truth[0];
    if (ignore != -1){
        if (truth[0] == ignore){
            space[0] = 0;
            return;
        }
    }
    if (scale != NULL){
        scale_x = scale[truth[0]];
    }
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        res += exp(data[i]-M);
    }
    space[0] = (-data[index]+M+log(res))*scale_x;
}

void crossentropy_gradient(float *data, int *truth, int num, float *scale, int ignore, float *space)
{
    float M = 0;
    float res = 0;
    float scale_x = 1;
    if (ignore != -1){
        if (truth[0] == ignore){
            space[0] = 0;
            return;
        }
    }
    if (scale != NULL){
        scale_x = scale[truth[0]];
    }
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        space[i] = exp(data[i]-M);
        res += space[i];
    }
    for (int i = 0; i < num; ++i){
        if (i == truth[0]){
            space[i] = (space[i]/res-1)*scale_x;
        } else {
            space[i] = space[i]/res*scale_x;
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

    l->workspace_size = l->inputs;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "CrossEntropy    Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_crossentropy_layer(Layer l, int num)
{
    if (!l.status){
        for (int i = 0; i < l.input_h*l.input_w; ++i){
            float *input = l.input+i*l.input_c;
            float *detect = l.detect+i*l.input_c;
            float *output = l.output+i;
            crossentropy(input, l.truth, l.input_c, l.scale, l.ignore, output);
            softmax(input, l.inputs, detect);
        }
    } else {
        for (int i = 0; i < num; ++i){
            for (int j = 0; j < l.input_h*l.input_w; ++j){
                float *input = l.input+(i*l.input_h*l.input_w+j)*l.input_c;
                int *truth = l.truth+i*l.output_h*l.output_w+j;
                float *output = l.output+i*l.output_h*l.output_w+j;
                crossentropy(input, truth, l.input_c, l.scale, l.ignore, output);
            }
        }
    }
    sum_cpu(l.output, l.outputs*num, l.loss);
    multy_cpu(l.loss, 1, (float)1/num, 1);
}

void backward_crossentropy_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        for (int j = 0; j < l.input_h*l.input_w; ++j){
            float *input = l.input+(i*l.input_h*l.input_w+j)*l.input_c;
            int *truth = l.truth+i*l.output_h*l.output_w+j;
            float *delta_l = l.delta+(i*l.input_h*l.input_w+j)*l.input_c;
            crossentropy_gradient(input, truth, l.input_c, l.scale, l.ignore, delta_l);
        }
    }
}

void zerograd_crossentropy_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
}
