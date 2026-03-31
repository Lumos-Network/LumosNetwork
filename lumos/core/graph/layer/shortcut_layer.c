#include "shortcut_layer.h"

Layer *make_shortcut_layer(Layer *shortcut, int shortcuttype, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = SHORTCUT;
    l->shortcut = shortcut;
    l->shortcut_type = shortcuttype;

    Activation type = load_activate_type(active);
    l->active = type;

    l->initialize = init_shortcut_layer;
    l->forward = forward_shortcut_layer;
    l->backward = backward_shortcut_layer;
    l->initializegpu = init_shortcut_layer_gpu;
    l->forwardgpu = forward_shortcut_layer_gpu;
    l->backwardgpu = backward_shortcut_layer_gpu;

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

    l->zerogradlayer = zerograd_shortcut_layer;
    l->zerogradlayergpu = zerograd_shortcut_layer_gpu;

    fprintf(stderr, "Shortcut        Layer    :    [active=%s]\n", active);
    return l;
}

void init_shortcut_layer(Layer *l, int w, int h, int c, int subdivision)
{
    Layer *shortcut = l->shortcut;
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    if (l->shortcut_type == SHORT){
        l->output_h = shortcut->input_h;
        l->output_w = shortcut->input_w;
        l->output_c = shortcut->input_c;
    } else {
        l->output_h = h;
        l->output_w = w;
        l->output_c = c;
    }
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = subdivision*l->outputs;
    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "Shortcut        Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_shortcut_layer(Layer l, int num)
{
    Layer *shortcut = l.shortcut;
    if (l.shortcut_type == SHORT){
        memcpy(l.output, shortcut->input, num*l.outputs*sizeof(float));
    } else {
        matrix_add_cpu(l.input, shortcut->input, num*l.inputs, l.output);
    }
    activate_list(l.output, num*l.outputs, l.output, l.active);
}

void backward_shortcut_layer(Layer l, int num, float *n_delta)
{
    Layer *shortcut = l.shortcut;
    if (l.shortcut_type == SHORT){
        memcpy(shortcut->delta, n_delta, num*l.outputs*sizeof(float));
    } else {
        gradient_list(l.output, num*l.outputs, l.workspace, l.active);
        matrix_multiply_cpu(l.workspace, n_delta, num*l.inputs, l.delta);
        memcpy(shortcut->delta, l.delta, num*l.inputs*sizeof(float));
    }
}

void zerograd_shortcut_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
}
