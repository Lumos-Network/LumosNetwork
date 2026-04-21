#include "convtranspose_layer.h"

Layer *make_convtranspose_layer(int filters, int ksize, int stride, int pad, int bias, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONVTRANSPOSE;
    l->filters = filters;
    l->ksize = ksize;
    l->stride = stride;
    l->pad = pad;
    l->bias = bias;

    Activation type = load_activate_type(active);
    l->active = type;

    l->initialize = init_convtranspose_layer;
    l->forward = forward_convtranspose_layer;
    l->backward = backward_convtranspose_layer;

    l->initializegpu = init_convtranspose_layer_gpu;
    l->forwardgpu = forward_convtranspose_layer_gpu;
    l->backwardgpu = backward_convtranspose_layer_gpu;

    l->sgdoptimizer = convtranspose_layer_SGDOptimizer;
    l->sgdoptimizergpu = convtranspose_layer_SGDOptimizer_gpu;

    l->weightinit = weightinit_convtranspose_layer;
    l->weightinitgpu = weightinit_convtranspose_layer_gpu;

    l->refresh = refresh_convtranspose_layer_weights;
    l->refreshgpu = refresh_convtranspose_layer_weights_gpu;

    l->saveweights = save_convtranspose_layer_weights;
    l->saveweightsgpu = save_convtranspose_layer_weights_gpu;

    l->zerogradlayer = zerograd_convtranspose_layer;
    l->zerogradlayergpu = zerograd_convtranspose_layer_gpu;

    fprintf(stderr, "ConvTranspose   Layer   :   [filters=%2d, ksize=%2d, stride=%2d, pad=%2d, bias=%d, active=%s]\n",
            l->filters, l->ksize, l->stride, l->pad, l->bias, active);
    return l;
}

void init_convtranspose_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = (l->input_h - 1)*l->stride + l->ksize - 2*l->pad;
    l->output_w = (l->input_w - 1)*l->stride + l->ksize - 2*l->pad;
    l->output_c = l->filters;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->ksize*l->ksize*l->input_c*l->output_h*l->output_w + l->filters*l->ksize*l->ksize*l->input_c;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));
    l->kernel_weights = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
    l->update_kernel_weights = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
    l.kernel_weights_delta = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
    if (l->bias){
        l->bias_weights = calloc(l->filters, sizeof(float));
        l->update_bias_weights = calloc(l->filters, sizeof(float));
        l->bias_delta = calloc(l->filters, sizeof(float));
        if (l->optimizer == SGD){
            l->momentum_bias_v = calloc(l->filters, sizeof(float));
            fill_cpu(l->momentum_bias_v, l->filters, 0, 1);
        }
    }
    if (l->optimizer == SGD){
        l->momentum_kernel_v = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
        fill_cpu(l->momentum_kernel_v, l->filters*l->ksize*l->ksize*l->input_c, 0, 1);
    }

    fprintf(stderr, "ConvTranspose   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_convtranspose_layer(Layer l, FILE *fp)
{
    if (fp){
        fread(l.kernel_weights, sizeof(float), l.filters*l.ksize*l.ksize*l.input_c, fp);
        memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
        if (l.bias){
            fread(l.bias_weights, sizeof(float), l.filters, fp);
            memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
        }
        return;
    }
    if (l.initcptkernel == NULL){
        convtranspose_kaiming_uniform_kernel_init(l, sqrt(5.0), "fan_in", "leaky_relu");
    } else {
        InitCptKernel initcptkernel = *l.initcptkernel;
        if (initcptkernel.initype == CONSTANT_I) convtranspose_constant_kernel_init(l, initcptkernel.x);
        else if (initcptkernel.initype == NORMAL_I) convtranspose_normal_kernel_init(l, initcptkernel.mean, initcptkernel.std);
        else if (initcptkernel.initype == UNIFORM_I) convtranspose_uniform_kernel_init(l, initcptkernel.min, initcptkernel.max);
        else if (initcptkernel.initype == XAVIER_NORMAL_I) convtranspose_xavier_normal_kernel_init(l, initcptkernel.a);
        else if (initcptkernel.initype == XAVIER_UNIFORM_I) convtranspose_xavier_uniform_kernel_init(l, initcptkernel.a);
        else if (initcptkernel.initype == KAIMING_NORMAL_I) convtranspose_kaiming_normal_kernel_init(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
        else if (initcptkernel.initype == KAIMING_UNIFORM_I) convtranspose_kaiming_uniform_kernel_init(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
        else convtranspose_kaiming_uniform_kernel_init(l, sqrt(5.0), "fan_in", "leaky_relu");
    }
    if (l.bias){
        if (l.initcptbias == NULL){
            convtranspose_kaiming_uniform_bias_init(l, "fan_in");
        } else {
            InitCptBias initcptbias = *l.initcptbias;
            if (initcptbias.initype == CONSTANT_I) convtranspose_constant_bias_init(l, initcptbias.x);
            else if (initcptbias.initype == NORMAL_I) convtranspose_normal_bias_init(l, initcptbias.mean, initcptbias.std);
            else if (initcptbias.initype == UNIFORM_I) convtranspose_uniform_bias_init(l, initcptbias.min, initcptbias.max);
            else if (initcptbias.initype == XAVIER_NORMAL_I) convtranspose_xavier_normal_bias_init(l, initcptbias.a);
            else if (initcptbias.initype == XAVIER_UNIFORM_I) convtranspose_xavier_uniform_bias_init(l, initcptbias.a);
            else if (initcptbias.initype == KAIMING_NORMAL_I) convtranspose_kaiming_normal_bias_init(l, initcptbias.mode);
            else if (initcptbias.initype == KAIMING_UNIFORM_I) convtranspose_kaiming_uniform_bias_init(l, initcptbias.mode);
            else convtranspose_kaiming_uniform_bias_init(l, "fan_in");
        }
    }
}

void forward_convtranspose_layer(Layer l, int num)
{
    
}

void backward_convtranspose_layer(Layer l, int num, float *n_delta);
void convtranspose_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize);
void refresh_convtranspose_layer_weights(Layer l);
void save_convtranspose_layer_weights(Layer l, FILE *fp);

void zerograd_convtranspose_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
    fill_cpu(l.kernel_weights_delta, l.filters*l.ksize*l.ksize*l.input_c, 0, 1);
    if (l.bias) fill_cpu(l.bias_delta, l.filters, 0, 1);
}

void convtranspose_constant_kernel_init(Layer l, float x)
{
    fill_cpu(l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c, x, 1);
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convtranspose_normal_kernel_init(Layer l, float mean, float std)
{
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = generate_normal(mean, std);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convtranspose_uniform_kernel_init(Layer l, float min, float max)
{
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = rand_uniform(min, max);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convtranspose_xavier_normal_kernel_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convtranspose_xavier_uniform_kernel_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    float a = sqrt(3.0) * std;
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = rand_uniform(-a, a);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convtranspose_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity)
{
    float fan = 0;
    float std = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) fan = l.ksize*l.ksize*l.filters;
    if (0 == strcmp(nonlinearity, "sigmoid")) a= 1;
    else if (0 == strcmp(nonlinearity, "tanh")) a = 5.0/3;
    else if (0 == strcmp(nonlinearity, "relu")) a = sqrt(2.0);
    else if (0 == strcmp(nonlinearity, "leaky_relu")){
        if (a == 0) a = 0.01;
        a = sqrt(2.0 / (1 + a*a));
    }
    else if (0 == strcmp(nonlinearity, "selu")) a = 3.0 / 4;
    std = a / sqrt(fan);
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convtranspose_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity)
{
    float fan = 0;
    float std = 0;
    float bound = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) fan = l.ksize*l.ksize*l.filters;
    if (0 == strcmp(nonlinearity, "sigmoid")) a= 1;
    else if (0 == strcmp(nonlinearity, "tanh")) a = 5.0/3;
    else if (0 == strcmp(nonlinearity, "relu")) a = sqrt(2.0);
    else if (0 == strcmp(nonlinearity, "leaky_relu")){
        if (a == 0) a = 0.01;
        a = sqrt(2.0 / (1 + a*a));
    }
    else if (0 == strcmp(nonlinearity, "selu")) a = 3.0 / 4;
    std = a / sqrt(fan);
    bound = sqrt(3.0) * std;
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = rand_uniform(-bound, bound);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convtranspose_constant_bias_init(Layer l, float x)
{
    fill_cpu(l.bias_weights, l.filters, x, 1);
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convtranspose_normal_bias_init(Layer l, float mean, float std)
{
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = generate_normal(mean, std);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convtranspose_uniform_bias_init(Layer l, float min, float max)
{
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = rand_uniform(min, max);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convtranspose_xavier_normal_bias_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convtranspose_xavier_uniform_bias_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    float a = sqrt(3.0) * std;
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = rand_uniform(-a, a);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convtranspose_kaiming_normal_bias_init(Layer l, char *mode)
{
    float fan = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) fan = l.ksize*l.ksize*l.filters;
    else fan = l.ksize*l.ksize*l.input_c;
    float std = 1 / sqrt(fan);
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = generate_normal(0, std);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convtranspose_kaiming_uniform_bias_init(Layer l, char *mode)
{
    float fan = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) fan = l.ksize*l.ksize*l.filters;
    else fan = l.ksize*l.ksize*l.input_c;
    float bound = 1 / sqrt(fan);
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = rand_uniform(-bound, bound);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}
