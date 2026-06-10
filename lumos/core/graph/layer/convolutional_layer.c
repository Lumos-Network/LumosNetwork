#include "convolutional_layer.h"

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int dilation, int bias, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONVOLUTIONAL;
    l->filters = filters;
    l->ksize = ksize;
    l->stride = stride;
    l->pad = pad;
    l->dilation = dilation;
    l->bias = bias;

    Activation type = load_activate_type(active);
    l->active = type;

    l->initialize = init_convolutional_layer;
    l->forward = forward_convolutional_layer;
    l->backward = backward_convolutional_layer;

    l->initializegpu = init_convolutional_layer_gpu;
    l->forwardgpu = forward_convolutional_layer_gpu;
    l->backwardgpu = backward_convolutional_layer_gpu;

    l->sgdoptimizer = convolutional_layer_SGDOptimizer;
    l->sgdoptimizergpu = convolutional_layer_SGDOptimizer_gpu;

    l->weightinit = weightinit_convolutional_layer;
    l->weightinitgpu = weightinit_convolutional_layer_gpu;

    l->refresh = refresh_convolutional_layer_weights;
    l->refreshgpu = refresh_convolutional_layer_weights_gpu;

    l->saveweights = save_convolutional_layer_weights;
    l->saveweightsgpu = save_convolutional_layer_weights_gpu;

    l->zerogradlayer = zerograd_convolutional_layer;
    l->zerogradlayergpu = zerograd_convolutional_layer_gpu;

    l->initcptkernel = NULL;
    l->initcptbias = NULL;

    fprintf(stderr, "Convolutional   Layer    :    [filters=%2d, ksize=%2d, stride=%2d, pad=%2d, bias=%d, active=%s]\n",
            l->filters, l->ksize, l->stride, l->pad, l->bias, active);
    return l;
}

void init_convolutional_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    int dksize = l->ksize*(l->dilation+1)-l->dilation;
    l->output_h = (l->input_h + 2 * l->pad - dksize) / l->stride + 1;
    l->output_w = (l->input_w + 2 * l->pad - dksize) / l->stride + 1;
    l->output_c = l->filters;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->ksize*l->ksize*l->input_c*l->output_h*l->output_w;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));
    l->kernel_weights = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
    l->update_kernel_weights = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
    l->kernel_weights_delta = calloc(l->filters*l->ksize*l->ksize*l->input_c, sizeof(float));
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

    fprintf(stderr, "Convolutional   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_convolutional_layer(Layer l, FILE *fp)
{
    if (fp){
        int flag = 0;
        int weights_num = l.filters*l.ksize*l.ksize*l.input_c;
        if (l.bias) weights_num += l.filters;
        float *weights = malloc(weights_num*sizeof(float));
        flag = fread(weights, sizeof(float), weights_num, fp);
        if (flag == weights_num){
            memcpy(l.kernel_weights, weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
            memcpy(l.update_kernel_weights, weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
            if (l.bias){
                memcpy(l.bias_weights, weights+l.filters*l.ksize*l.ksize*l.input_c, l.filters*sizeof(float));
                memcpy(l.update_bias_weights, weights+l.filters*l.ksize*l.ksize*l.input_c, l.filters*sizeof(float));
            }
            return;
        }
        free(weights);
    }
    if (l.initcptkernel == NULL){
        convolutional_kaiming_uniform_kernel_init(l, sqrt(5.0), "fan_in", "leaky");
    } else {
        InitCptKernel initcptkernel = *l.initcptkernel;
        if (initcptkernel.initype == CONSTANT_I) convolutional_constant_kernel_init(l, initcptkernel.x);
        else if (initcptkernel.initype == NORMAL_I) convolutional_normal_kernel_init(l, initcptkernel.mean, initcptkernel.std);
        else if (initcptkernel.initype == UNIFORM_I) convolutional_uniform_kernel_init(l, initcptkernel.min, initcptkernel.max);
        else if (initcptkernel.initype == XAVIER_NORMAL_I) convolutional_xavier_normal_kernel_init(l, initcptkernel.a);
        else if (initcptkernel.initype == XAVIER_UNIFORM_I) convolutional_xavier_uniform_kernel_init(l, initcptkernel.a);
        else if (initcptkernel.initype == KAIMING_NORMAL_I) convolutional_kaiming_normal_kernel_init(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
        else if (initcptkernel.initype == KAIMING_UNIFORM_I) convolutional_kaiming_uniform_kernel_init(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
        else convolutional_kaiming_uniform_kernel_init(l, sqrt(5.0), "fan_in", "leaky");
    }
    if (l.bias){
        if (l.initcptbias == NULL){
            convolutional_kaiming_uniform_bias_init(l, "fan_in");
        } else {
            InitCptBias initcptbias = *l.initcptbias;
            if (initcptbias.initype == CONSTANT_I) convolutional_constant_bias_init(l, initcptbias.x);
            else if (initcptbias.initype == NORMAL_I) convolutional_normal_bias_init(l, initcptbias.mean, initcptbias.std);
            else if (initcptbias.initype == UNIFORM_I) convolutional_uniform_bias_init(l, initcptbias.min, initcptbias.max);
            else if (initcptbias.initype == XAVIER_NORMAL_I) convolutional_xavier_normal_bias_init(l, initcptbias.a);
            else if (initcptbias.initype == XAVIER_UNIFORM_I) convolutional_xavier_uniform_bias_init(l, initcptbias.a);
            else if (initcptbias.initype == KAIMING_NORMAL_I) convolutional_kaiming_normal_bias_init(l, initcptbias.mode);
            else if (initcptbias.initype == KAIMING_UNIFORM_I) convolutional_kaiming_uniform_bias_init(l, initcptbias.mode);
            else convolutional_kaiming_uniform_bias_init(l, "fan_in");
        }
    }
}

void forward_convolutional_layer(Layer l, int num)
{
    fill_cpu(l.output, num*l.outputs, 0, 1);
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.dilation, l.workspace);
        gemm(0, 0, l.filters, l.output_w*l.output_h, l.ksize*l.ksize*l.input_c, 1,
            l.kernel_weights, l.ksize*l.ksize*l.input_c, l.workspace, l.output_w*l.output_h,
            1, output, l.output_w*l.output_h);
    }
    if (l.bias){
        add_bias(l.output, l.bias_weights, num, l.filters, l.output_h*l.output_w);
    }
    if (l.active == LINEAR) return;
    activate_list(l.output, num*l.outputs, l.output, l.active);
}

void backward_convolutional_layer(Layer l, int num, float *n_delta)
{
    gradient_list(l.output, num*l.outputs, n_delta, l.active);
    if (l.bias){
        backward_bias(l.bias_delta, n_delta, num, l.filters, l.output_h*l.output_w);
    }
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.dilation, l.workspace);
        gemm(0, 1, l.filters, l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, 1,
            delta_n, l.output_w*l.output_h, l.workspace, l.output_w*l.output_h, 1,
            l.kernel_weights_delta, l.ksize*l.ksize*l.input_c);
        gemm(1, 0, l.ksize*l.ksize*l.input_c, l.output_h*l.output_w, l.filters, 1,
            l.kernel_weights, l.ksize*l.ksize*l.input_c, delta_n, l.output_w*l.output_h,
            0, l.workspace, l.output_w*l.output_h);
        col2im(l.workspace, l.ksize, l.stride, l.pad, l.dilation, l.input_h, l.input_w, l.input_c, delta_l);
    }
}

void convolutional_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize)
{
    float *momentum_kernel_v;
    float *momentum_bias_v;
    if (decay != 0){
        saxpy_cpu(l.kernel_weights_delta, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c, decay, l.kernel_weights_delta);
    }
    if (momentum != 0){
        multy_cpu(l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c, momentum, 1);
        saxpy_cpu(l.momentum_kernel_v, l.kernel_weights_delta, l.filters*l.ksize*l.ksize*l.input_c, 1-dampening, l.momentum_kernel_v);
        if (nesterov){
            saxpy_cpu(l.kernel_weights_delta, l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c, momentum, l.kernel_weights_delta);
            momentum_kernel_v = l.kernel_weights_delta;
        } else {
            momentum_kernel_v = l.momentum_kernel_v;
        }
    }
    if (l.bias){
        if (decay != 0){
            saxpy_cpu(l.bias_delta, l.update_bias_weights, l.filters, decay, l.bias_delta);
        }
        if (momentum != 0){
            multy_cpu(l.momentum_bias_v, l.filters, momentum, 1);
            saxpy_cpu(l.momentum_bias_v, l.bias_delta, l.filters, 1-dampening, l.momentum_bias_v);
            if (nesterov){
                saxpy_cpu(l.bias_delta, l.momentum_bias_v, l.filters, momentum, l.bias_delta);
                momentum_bias_v = l.bias_delta;
            } else {
                momentum_bias_v = l.momentum_bias_v;
            }
        }
    }
    if (maximize){
        saxpy_cpu(l.update_kernel_weights, momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c, -rate, l.update_kernel_weights);
        if (l.bias) saxpy_cpu(l.update_bias_weights, momentum_bias_v, l.filters, -rate, l.update_bias_weights);
    } else {
        saxpy_cpu(l.update_kernel_weights, momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c, rate, l.update_kernel_weights);
        if (l.bias) saxpy_cpu(l.update_bias_weights, momentum_bias_v, l.filters, rate, l.update_bias_weights);
    }
}

void refresh_convolutional_layer_weights(Layer l)
{
    memcpy(l.kernel_weights, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
    if (l.bias){
        memcpy(l.bias_weights, l.update_bias_weights, l.filters*sizeof(float));
    }
}

void save_convolutional_layer_weights(Layer l, FILE *fp)
{
    fwrite(l.kernel_weights, sizeof(float), l.ksize*l.ksize*l.filters*l.input_c, fp);
    if (l.bias){
        fwrite(l.bias_weights, sizeof(float), l.filters, fp);
    }
}

void zerograd_convolutional_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
    fill_cpu(l.kernel_weights_delta, l.filters*l.ksize*l.ksize*l.input_c, 0, 1);
    if (l.bias) fill_cpu(l.bias_delta, l.filters, 0, 1);
}

void convolutional_constant_kernel_init(Layer l, float x)
{
    fill_cpu(l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c, x, 1);
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convolutional_normal_kernel_init(Layer l, float mean, float std)
{
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = generate_normal(mean, std);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convolutional_uniform_kernel_init(Layer l, float min, float max)
{
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = rand_uniform(min, max);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convolutional_xavier_normal_kernel_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float));
}

void convolutional_xavier_uniform_kernel_init(Layer l, float gain)
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

void convolutional_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity)
{
    float fan = 0;
    float std = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) fan = l.ksize*l.ksize*l.filters;
    if (0 == strcmp(nonlinearity, "sigmoid")) a= 1;
    else if (0 == strcmp(nonlinearity, "tanh")) a = 5.0/3;
    else if (0 == strcmp(nonlinearity, "relu")) a = sqrt(2.0);
    else if (0 == strcmp(nonlinearity, "leaky")){
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

void convolutional_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity)
{
    float fan = 0;
    float std = 0;
    float bound = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) fan = l.ksize*l.ksize*l.filters;
    if (0 == strcmp(nonlinearity, "sigmoid")) a= 1;
    else if (0 == strcmp(nonlinearity, "tanh")) a = 5.0/3;
    else if (0 == strcmp(nonlinearity, "relu")) a = sqrt(2.0);
    else if (0 == strcmp(nonlinearity, "leaky")){
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

void convolutional_constant_bias_init(Layer l, float x)
{
    fill_cpu(l.bias_weights, l.filters, x, 1);
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convolutional_normal_bias_init(Layer l, float mean, float std)
{
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = generate_normal(mean, std);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convolutional_uniform_bias_init(Layer l, float min, float max)
{
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = rand_uniform(min, max);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convolutional_xavier_normal_bias_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convolutional_xavier_uniform_bias_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters; ++i){
        l.bias_weights[i] = rand_uniform(-std, std);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.filters*sizeof(float));
}

void convolutional_kaiming_normal_bias_init(Layer l, char *mode)
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

void convolutional_kaiming_uniform_bias_init(Layer l, char *mode)
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
