#include "local_layer.h"

Layer *make_local_layer(int filters, int ksize, int stride, int pad, int dilation, int bias, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = LOCAL;
    l->filters = filters;
    l->ksize = ksize;
    l->stride = stride;
    l->pad = pad;
    l->dilation = dilation;
    l->bias = 0;

    Activation type = load_activate_type(active);
    l->active = type;

    l->initialize = init_local_layer;
    l->forward = forward_local_layer;
    l->backward = backward_local_layer;

    l->initializegpu = init_local_layer_gpu;
    l->forwardgpu = forward_local_layer_gpu;
    l->backwardgpu = backward_local_layer_gpu;

    l->sgdoptimizer = local_layer_SGDOptimizer;
    l->sgdoptimizergpu = local_layer_SGDOptimizer_gpu;

    l->weightinit = weightinit_local_layer;
    l->weightinitgpu = weightinit_local_layer_gpu;

    l->refresh = refresh_local_layer_weights;
    l->refreshgpu = refresh_local_layer_weights_gpu;

    l->saveweights = save_local_layer_weights;
    l->saveweightsgpu = save_local_layer_weights_gpu;

    l->zerogradlayer = zerograd_local_layer;
    l->zerogradlayergpu = zerograd_local_layer_gpu;

    l->initcptkernel = NULL;
    l->initcptbias = NULL;

    fprintf(stderr, "local   Layer    :    [filters=%2d, ksize=%2d, stride=%2d, pad=%2d, bias=%d, active=%s]\n",
            l->filters, l->ksize, l->stride, l->pad, l->bias, active);
    return l;
}

void init_local_layer(Layer *l, int w, int h, int c, int subdivision)
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
    l->kernel_weights = calloc(l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w, sizeof(float));
    l->update_kernel_weights = calloc(l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w, sizeof(float));
    l->kernel_weights_delta = calloc(l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w, sizeof(float));
    if (l->optimizer == SGD){
        l->momentum_kernel_v = calloc(l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w, sizeof(float));
        fill_cpu(l->momentum_kernel_v, l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w, 0, 1);
    }

    fprintf(stderr, "local   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_local_layer(Layer l, FILE *fp)
{
    if (fp){
        int flag = 0;
        int weights_num = l.filters*l.ksize*l.ksize*l.input_c*l.output_h*l.output_w;
        float *weights = malloc(weights_num*sizeof(float));
        flag = fread(weights, sizeof(float), weights_num, fp);
        if (flag == weights_num){
            memcpy(l.kernel_weights, weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_h*l.output_w*sizeof(float));
            memcpy(l.update_kernel_weights, weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_h*l.output_w*sizeof(float));
            return;
        }
        free(weights);
    }
    if (l.initcptkernel == NULL){
        local_kaiming_uniform_kernel_init(l, sqrt(5.0), "fan_in", "leaky");
    } else {
        InitCptKernel initcptkernel = *l.initcptkernel;
        if (initcptkernel.initype == CONSTANT_I) local_constant_kernel_init(l, initcptkernel.x);
        else if (initcptkernel.initype == NORMAL_I) local_normal_kernel_init(l, initcptkernel.mean, initcptkernel.std);
        else if (initcptkernel.initype == UNIFORM_I) local_uniform_kernel_init(l, initcptkernel.min, initcptkernel.max);
        else if (initcptkernel.initype == XAVIER_NORMAL_I) local_xavier_normal_kernel_init(l, initcptkernel.a);
        else if (initcptkernel.initype == XAVIER_UNIFORM_I) local_xavier_uniform_kernel_init(l, initcptkernel.a);
        else if (initcptkernel.initype == KAIMING_NORMAL_I) local_kaiming_normal_kernel_init(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
        else if (initcptkernel.initype == KAIMING_UNIFORM_I) local_kaiming_uniform_kernel_init(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
        else local_kaiming_uniform_kernel_init(l, sqrt(5.0), "fan_in", "leaky");
    }
}

void forward_local_layer(Layer l, int num)
{
    fill_cpu(l.output, num*l.outputs, 0, 1);
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.dilation, l.workspace);
        for (int j = 0; j < l.output_w*l.output_h; ++j){
            float *weight = l.kernel_weights + j*l.ksize*l.ksize*l.input_c*l.filters;
            float *data = l.workspace + j;
            float *out = output + j;
            gemm(0, 0, l.filters, 1, l.ksize*l.ksize*l.input_c, 1,
                weight, l.ksize*l.ksize*l.input_c, data, l.output_w*l.output_h,
                1, out, l.output_w*l.output_h);
        }
    }
    if (l.active == LINEAR) return;
    activate_list(l.output, num*l.outputs, l.output, l.active);
}

void backward_local_layer(Layer l, int num, float *n_delta)
{
    gradient_list(l.output, num*l.outputs, n_delta, l.active);
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        im2col(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.dilation, l.workspace);
        for (int j = 0; j < l.output_w*l.output_h; ++j){
            float *delta = delta_n + j;
            float *data = l.workspace + j;
            float *weight_delta = l.kernel_weights_delta + j*l.ksize*l.ksize*l.input_c*l.filters;
            gemm(0, 1, l.filters, l.ksize*l.ksize*l.input_c, 1, 1,
                delta, l.output_w*l.output_h, data, l.output_w*l.output_h, 1,
                weight_delta, l.ksize*l.ksize*l.input_c);
            float *weight = l.kernel_weights + j*l.ksize*l.ksize*l.input_c*l.filters;
            gemm(1, 0, l.ksize*l.ksize*l.input_c, 1, l.filters, 1,
                weight, l.ksize*l.ksize*l.input_c, delta, l.output_w*l.output_h, 0,
                data, l.output_w*l.output_h);
        }
        col2im(l.workspace, l.ksize, l.stride, l.pad, l.dilation, l.input_h, l.input_w, l.input_c, delta_l);
    }
}

void local_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize)
{
    float *momentum_kernel_v;
    if (decay != 0){
        saxpy_cpu(l.kernel_weights_delta, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, decay, l.kernel_weights_delta);
    }
    if (momentum != 0){
        multy_cpu(l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, momentum, 1);
        saxpy_cpu(l.momentum_kernel_v, l.kernel_weights_delta, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, 1-dampening, l.momentum_kernel_v);
        if (nesterov){
            saxpy_cpu(l.kernel_weights_delta, l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, momentum, l.kernel_weights_delta);
            momentum_kernel_v = l.kernel_weights_delta;
        } else {
            momentum_kernel_v = l.momentum_kernel_v;
        }
    }
    if (maximize){
        saxpy_cpu(l.update_kernel_weights, momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, -rate, l.update_kernel_weights);
    } else {
        saxpy_cpu(l.update_kernel_weights, momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, rate, l.update_kernel_weights);
    }
}

void refresh_local_layer_weights(Layer l)
{
    memcpy(l.kernel_weights, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float));
}

void save_local_layer_weights(Layer l, FILE *fp)
{
    fwrite(l.kernel_weights, sizeof(float), l.ksize*l.ksize*l.filters*l.input_c*l.output_w*l.output_h, fp);
}

void zerograd_local_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
    fill_cpu(l.kernel_weights_delta, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, 0, 1);
}

void local_constant_kernel_init(Layer l, float x)
{
    fill_cpu(l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, x, 1);
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float));
}

void local_normal_kernel_init(Layer l, float mean, float std)
{
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        l.kernel_weights[i] = generate_normal(mean, std);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float));
}

void local_uniform_kernel_init(Layer l, float min, float max)
{
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        l.kernel_weights[i] = rand_uniform(min, max);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float));
}

void local_xavier_normal_kernel_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        l.kernel_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float));
}

void local_xavier_uniform_kernel_init(Layer l, float gain)
{
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    float a = sqrt(3.0) * std;
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        l.kernel_weights[i] = rand_uniform(-a, a);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float));
}

void local_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity)
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
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        l.kernel_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float));
}

void local_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity)
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
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        l.kernel_weights[i] = rand_uniform(-bound, bound);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float));
}
