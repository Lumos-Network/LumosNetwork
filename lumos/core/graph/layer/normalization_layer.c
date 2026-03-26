#include "normalization_layer.h"

Layer *make_normalization_layer(float momentum, int affine, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = NORMALIZE;
    l->bn_momentum = momentum;
    l->affine = affine;

    Activation type = load_activate_type(active);
    l->active = type;

    l->initialize = init_normalization_layer;
    l->forward = forward_normalization_layer;
    l->backward = backward_normalization_layer;
    l->update = NULL;
    l->initializegpu = init_normalization_layer_gpu;
    l->forwardgpu = forward_normalization_layer_gpu;
    l->backwardgpu = backward_normalization_layer_gpu;
    l->updategpu = NULL;

    l->sgdoptimizer = normalization_layer_SGDOptimizer;
    l->sgdoptimizergpu = normalization_layer_SGDOptimizer_gpu;

    l->weightinit = weightinit_normalization_layer;
    l->weightinitgpu = weightinit_normalization_layer_gpu;

    l->refresh = refresh_normalization_layer_weights;
    l->refreshgpu = refresh_normalization_layer_weights_gpu;

    l->saveweights = save_normalization_layer_weights;
    l->saveweightsgpu = save_normalization_layer_weights_gpu;

    l->zerogradlayer = zerograd_normalization_layer;
    l->zerogradlayergpu = zerograd_normalization_layer_gpu;

    fprintf(stderr, "Normalization   Layer    :    [momentum=%.1f, affine=%d, bias=%d, active=%s]\n",
            l->bn_momentum, l->affine, l->bias, active);
    return l;
}

void init_normalization_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->filters = c;
    l->ksize = h*w;

    l->workspace_size = subdivision*l->outputs;

    l->mean = calloc(l->filters, sizeof(float));
    l->variance = calloc(l->filters, sizeof(float));
    l->rolling_mean = calloc(l->filters, sizeof(float));
    l->rolling_variance = calloc(l->filters, sizeof(float));
    l->mean_delta = calloc(l->filters, sizeof(float));
    l->variance_delta = calloc(l->filters, sizeof(float));
    l->kernel_weights = calloc(l->filters, sizeof(float));
    l->bias_weights = calloc(l->filters, sizeof(float));
    if (l->affine){
        l->update_kernel_weights = calloc(l->filters, sizeof(float));
        l->update_bias_weights = calloc(l->filters, sizeof(float));
        if (l->optimizer == SGD){
            l->momentum_kernel_v = calloc(l->filters, sizeof(float));
            fill_cpu(l->momentum_kernel_v, l->filters, 0, 1);
            l->momentum_bias_v = calloc(l->filters, sizeof(float));
            fill_cpu(l->momentum_bias_v, l->filters, 0, 1);
        }
    }
    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));
    l->norm_x = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "Normalization   Layer\n");
}

void weightinit_normalization_layer(Layer l, FILE *fp)
{
    if (fp){
        fread(l.kernel_weights, sizeof(float), l.filters, fp);
        fread(l.bias_weights, sizeof(float), l.filters, fp);
        fread(l.rolling_mean, sizeof(float), l.filters, fp);
        fread(l.rolling_variance, sizeof(float), l.filters, fp);
        if (l.affine){
            memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*sizeof(float));
            memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
        }
        return;
    }
    fill_cpu(l.kernel_weights, l.filters, 1, 1);
    fill_cpu(l.bias_weights, l.filters, 0, 1);
    if (l.affine){
        fill_cpu(l.update_kernel_weights, l.filters, 1, 1);
        fill_cpu(l.update_bias_weights, l.filters, 0, 1);
    }
}

void forward_normalization_layer(Layer l, int num)
{
    if (l.status){
        normalize_mean(l.input, l.ksize, l.filters, num, l.mean);
        normalize_variance(l.input, l.ksize, l.filters, num, l.mean, l.variance);
        multy_cpu(l.rolling_mean, l.filters, 1-l.bn_momentum, 1);
        multy_cpu(l.rolling_variance, l.filters, 1-l.bn_momentum, 1);
        saxpy_cpu(l.rolling_mean, l.mean, l.filters, l.bn_momentum, l.rolling_mean);
        saxpy_cpu(l.rolling_variance, l.variance, l.filters, l.bn_momentum, l.rolling_variance);
    }
    for (int i = 0; i < num; ++i){
        float *input = l.input + l.inputs*i;
        float *output = l.output + l.outputs*i;
        float *norm_x = l.norm_x + l.inputs*i;
        if (l.status) normalize_cpu(input, l.mean, l.variance, l.ksize, l.filters, output);
        if (!l.status) normalize_cpu(input, l.rolling_mean, l.rolling_variance, l.ksize, l.filters, output);
        memcpy(norm_x, output, l.outputs*sizeof(float));
        scale_bias(output, l.kernel_weights, l.filters, l.ksize);
        add_bias(output, l.bias_weights, l.filters, l.ksize);
    }
    activate_list(l.output, num*l.outputs, l.output, l.active);
}

void backward_normalization_layer(Layer l, int num, float *n_delta)
{
    gradient_list(l.output, num*l.outputs, l.workspace, l.active);
    matrix_multiply_cpu(n_delta, l.workspace, num*l.outputs, n_delta);
    memcpy(l.delta, n_delta, num*l.inputs*sizeof(float));
    for (int i = 0; i < num; ++i){
        float *input = l.input + i*l.inputs;
        float *delta_l = l.delta + i*l.inputs;
        float *delta_n = n_delta + i*l.outputs;
        float *norm_x = l.norm_x + i*l.inputs;
        scale_bias(delta_l, l.kernel_weights, l.filters, l.ksize);
        gradient_normalize_mean(delta_l, l.variance, l.ksize, l.filters, l.mean_delta);
        gradient_normalize_variance(delta_l, input, l.mean, l.variance, l.ksize, l.filters, l.variance_delta);
        gradient_normalize_cpu(input, l.mean, l.variance, l.mean_delta, l.variance_delta, l.ksize, l.filters, delta_l, delta_l);
        gradient_scale(norm_x, l.mean, l.variance, delta_n, l.ksize, l.filters, l.workspace);
        saxpy_cpu(l.kernel_weights_delta, l.workspace, l.filters, 1./num, l.kernel_weights_delta);
        gradient_bias(delta_n, l.ksize, l.filters, l.workspace);
        saxpy_cpu(l.bias_delta, l.workspace, l.filters, 1./num, l.bias_delta);
    }
}

void normalization_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize, float *n_delta)
{
    float *momentum_kernel_v;
    float *momentum_bias_v;
    if (decay != 0){
        saxpy_cpu(l.kernel_weights_delta, l.update_kernel_weights, l.filters, 1-decay, l.workspace);
    }
    if (momentum != 0){
        multy_cpu(l.momentum_kernel_v, l.filters, momentum, 1);
        saxpy_cpu(l.momentum_kernel_v, l.workspace, l.filters, 1-dampening, l.momentum_kernel_v);
        if (nesterov){
            saxpy_cpu(l.workspace, l.momentum_kernel_v, l.filters, momentum, l.workspace);
            momentum_kernel_v = l.workspace;
        } else {
            momentum_kernel_v = l.momentum_kernel_v;
        }
    }
    if (decay != 0){
        saxpy_cpu(l.bias_delta, l.update_bias_weights, l.filters, 1-decay, l.workspace);
    }
    if (momentum != 0){
        multy_cpu(l.momentum_bias_v, l.filters, momentum, 1);
        saxpy_cpu(l.momentum_bias_v, l.workspace, l.filters, 1-dampening, l.momentum_bias_v);
        if (nesterov){
            saxpy_cpu(l.workspace, l.momentum_bias_v, l.filters, momentum, l.workspace);
            momentum_bias_v = l.workspace;
        } else {
            momentum_bias_v = l.momentum_bias_v;
        }
    }
    if (maximize){
        saxpy_cpu(l.update_kernel_weights, momentum_kernel_v, l.filters, -rate, l.update_kernel_weights);
        saxpy_cpu(l.update_bias_weights, momentum_bias_v, l.filters, -rate, l.update_bias_weights);
    } else {
        saxpy_cpu(l.update_kernel_weights, momentum_kernel_v, l.filters, rate, l.update_kernel_weights);
        saxpy_cpu(l.update_bias_weights, momentum_bias_v, l.filters, rate, l.update_bias_weights);
    }
}

void refresh_normalization_layer_weights(Layer l)
{
    if (l.affine){
        memcpy(l.kernel_weights, l.update_kernel_weights, l.filters*sizeof(float));
        memcpy(l.bias_weights, l.update_bias_weights, l.filters*sizeof(float));
    }
}

void save_normalization_layer_weights(Layer l, FILE *fp)
{
    fwrite(l.kernel_weights, sizeof(float), l.filters, fp);
    fwrite(l.bias_weights, sizeof(float), l.filters, fp);
    fwrite(l.rolling_mean, sizeof(float), l.filters, fp);
    fwrite(l.rolling_variance, sizeof(float), l.filters, fp);
}

void zerograd_normalization_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
    fill_cpu(l.kernel_weights_delta, l.filters, 0, 1);
    fill_cpu(l.bias_delta, l.filters, 0, 1);
}
