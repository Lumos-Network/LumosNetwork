#include "nll_layer.h"

Layer *make_nll_layer(int group)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = NLL;
    l->group = group;

    l->initialize = init_nll_layer;
    l->forward = forward_nll_layer;
    l->backward = backward_nll_layer;

    l->initializegpu = init_nll_layer_gpu;
    l->forwardgpu = forward_nll_layer_gpu;
    l->backwardgpu = backward_nll_layer_gpu;

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

    l->zerogradlayer = zerograd_nll_layer;
    l->zerogradlayergpu = zerograd_nll_layer_gpu;

    fprintf(stderr, "NLL             Layer    :    [output=%4d]\n", 1);
    return l;
}

void init_nll_layer(Layer *l, int w, int h, int c, int subdivision)
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

    fprintf(stderr, "NLL             Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_nll_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *output = l.output+offset_o;
        float *truth = l.truth+offset_t;
        for (int j = 0; j < l.group; ++j){
            if (truth[j] == 1) output[0] = -log(input[j]);
        }
    }
    sum_cpu(l.output, l.outputs*num, l.loss);
    multy_cpu(l.loss, 1, (float)1/num, 1);
}

void backward_nll_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *delta_l = l.delta+offset_i;
        float *truth = l.truth+offset_t;
        for (int j = 0; j < l.group; ++j){
            if (truth[j] == 1) delta_l[j] = -1/(input[j]*num);
            else delta_l[j] = 0;
        }
    }
}

void zerograd_nll_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
}
