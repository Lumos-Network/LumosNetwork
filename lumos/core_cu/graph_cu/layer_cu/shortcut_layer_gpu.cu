#include "shortcut_layer_gpu.h"

void init_shortcut_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
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
    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "Shortcut        Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_shortcut_layer_gpu(Layer l, int num)
{
    Layer *shortcut = l.shortcut;
    if (l.shortcut_type == SHORT){
        cudaMemcpy(l.output, shortcut->input, num*l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        matrix_add_gpu(l.input, shortcut->input, num*l.inputs, l.output);
    }
    activate_list_gpu(l.output, num*l.outputs, l.output, l.active);
}

void backward_shortcut_layer_gpu(Layer l, int num, float *n_delta)
{
    Layer *shortcut = l.shortcut;
    if (l.shortcut_type == SHORT){
        cudaMemcpy(shortcut->delta, n_delta, num*l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        gradient_list_gpu(l.output, num*l.outputs, l.workspace, l.active);
        matrix_multiply_gpu(l.workspace, n_delta, num*l.inputs, l.delta);
        cudaMemcpy(shortcut->delta, l.delta, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void zerograd_shortcut_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
}
