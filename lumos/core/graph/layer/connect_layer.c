#include "connect_layer.h"

void denom_cpu(float *data, float x, float eps, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = sqrt(data[i]) / x + eps;
    }
}

Layer *make_connect_layer(int output, int bias, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = CONNECT;
    l->ksize = output;
    l->bias = bias;

    Activation type = load_activate_type(active);
    l->active = type;

    l->initialize = init_connect_layer;
    l->forward = forward_connect_layer;
    l->backward = backward_connect_layer;
    l->update = update_connect_layer;

    l->initializegpu = init_connect_layer_gpu;
    l->forwardgpu = forward_connect_layer_gpu;
    l->backwardgpu = backward_connect_layer_gpu;
    l->updategpu = update_connect_layer_gpu;

    l->sgdoptimizer = connect_layer_SGDOptimizer;
    l->sgdoptimizergpu = connect_layer_SGDOptimizer_gpu;

    l->adamoptimizer = connect_layer_AdamOptimizer;
    // l->adamoptimizergpu = connect_layer_AdamOptimizer_gpu;

    l->weightinit = weightinit_connect_layer;
    l->weightinitgpu = weightinit_connect_layer_gpu;

    l->refresh = refresh_connect_layer_weights;
    l->refreshgpu = refresh_connect_layer_weights_gpu;

    l->saveweights = save_connect_layer_weights;
    l->saveweightsgpu = save_connect_layer_weights_gpu;

    l->zerogradlayer = zerograd_connect_layer;
    l->zerogradlayergpu = zerograd_connect_layer_gpu;

    fprintf(stderr, "Connect         Layer    :    [output=%4d, bias=%d, active=%s]\n", l->ksize, l->bias, active);
    return l;
}

void init_connect_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = 1;
    l->input_w = 1;
    l->input_c = h*w*c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = l->ksize;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->inputs * l->outputs;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));
    l->kernel_weights = calloc(l->inputs * l->outputs, sizeof(float));
    l->update_kernel_weights = calloc(l->inputs * l->outputs, sizeof(float));
    l->kernel_weights_delta = calloc(l->inputs * l->outputs, sizeof(float));
    if (l->bias){
        l->bias_weights = calloc(l->outputs, sizeof(float));
        l->update_bias_weights = calloc(l->outputs, sizeof(float));
        l->bias_delta= calloc(l->outputs, sizeof(float));
        if (l->optimizer == SGD){
            l->momentum_bias_v = calloc(l->outputs, sizeof(float));
            fill_cpu(l->momentum_bias_v, l->outputs, 0, 1);
        }
        if (l->optimizer == ADAM){
            l->exp_avg_bias = calloc(l->outputs, sizeof(float));
            l->exp_avg_sq_bias = calloc(l->outputs, sizeof(float));
            l->exp_avg_sq_bias_max = calloc(l->outputs, sizeof(float));
            fill_cpu(l->exp_avg_bias, l->outputs, 0, 1);
            fill_cpu(l->exp_avg_sq_bias, l->outputs, 0, 1);
            fill_cpu(l->exp_avg_sq_bias_max, l->outputs, 0, 1);
        }
    }
    if (l->optimizer == SGD){
        l->momentum_kernel_v = calloc(l->inputs*l->outputs, sizeof(float));
        fill_cpu(l->momentum_kernel_v, l->inputs*l->outputs, 0, 1);
    }
    if (l->optimizer == ADAM){
        l->step_t = 0;
        l->exp_avg_kernel = calloc(l->inputs*l->outputs, sizeof(float));
        l->exp_avg_sq_kernel = calloc(l->inputs*l->outputs, sizeof(float));
        l->exp_avg_sq_kernel_max = calloc(l->inputs*l->outputs, sizeof(float));
        fill_cpu(l->exp_avg_kernel, l->inputs*l->outputs, 0, 1);
        fill_cpu(l->exp_avg_sq_kernel, l->inputs*l->outputs, 0, 1);
        fill_cpu(l->exp_avg_sq_kernel_max, l->inputs*l->outputs, 0, 1);
    }

    fprintf(stderr, "Connect         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_connect_layer(Layer l, FILE *fp)
{
    if (fp){
        fread(l.kernel_weights, sizeof(float), l.outputs*l.inputs, fp);
        memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
        if (l.bias){
            fread(l.bias_weights, sizeof(float), l.outputs, fp);
            memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
        }
        return;
    }
    InitCptKernel initcptkernel = *l.initcptkernel;
    if (initcptkernel.initype == CONSTANT_I) connect_constant_kernel_init(l, initcptkernel.x);
    else if (initcptkernel.initype == NORMAL_I) connect_normal_kernel_init(l, initcptkernel.mean, initcptkernel.std);
    else if (initcptkernel.initype == UNIFORM_I) connect_uniform_kernel_init(l, initcptkernel.min, initcptkernel.max);
    else if (initcptkernel.initype == XAVIER_NORMAL_I) connect_xavier_normal_kernel_init(l, initcptkernel.a);
    else if (initcptkernel.initype == XAVIER_UNIFORM_I) connect_xavier_uniform_kernel_init(l, initcptkernel.a);
    else if (initcptkernel.initype == KAIMING_NORMAL_I) connect_kaiming_normal_kernel_init(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
    else if (initcptkernel.initype == KAIMING_UNIFORM_I) connect_kaiming_uniform_kernel_init(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
    else connect_kaiming_uniform_kernel_init(l, sqrt(5.0), "fan_in", "leaky_relu");
    if (l.bias){
        InitCptBias initcptbias = *l.initcptbias;
        if (initcptbias.initype == CONSTANT_I) connect_constant_bias_init(l, initcptbias.x);
        else if (initcptbias.initype == NORMAL_I) connect_normal_bias_init(l, initcptbias.mean, initcptbias.std);
        else if (initcptbias.initype == UNIFORM_I) connect_uniform_bias_init(l, initcptbias.min, initcptbias.max);
        else if (initcptbias.initype == XAVIER_NORMAL_I) connect_xavier_normal_bias_init(l, initcptbias.a);
        else if (initcptbias.initype == XAVIER_UNIFORM_I) connect_xavier_uniform_bias_init(l, initcptbias.a);
        else if (initcptbias.initype == KAIMING_NORMAL_I) connect_kaiming_normal_bias_init(l, initcptbias.mode);
        else if (initcptbias.initype == KAIMING_UNIFORM_I) connect_kaiming_uniform_bias_init(l, initcptbias.mode);
        else connect_kaiming_uniform_bias_init(l, "fan_in");
    }
}

void forward_connect_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        gemm(0, 0, l.outputs, l.inputs, l.inputs, 1,
             1, l.kernel_weights, input, output);
        if (l.bias){
            add_bias(output, l.bias_weights, l.ksize, 1);
        }
    }
    activate_list(l.output, num*l.outputs, l.output, l.active);
}

void backward_connect_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list(output, l.outputs, l.workspace, l.active);
        matrix_multiply_cpu(delta_n, l.workspace, l.outputs, delta_n);
        gemm(1, 0, l.output_c, l.input_c, l.output_c, l.input_w, 1,
             l.kernel_weights, delta_n, delta_l);
        gemm(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.workspace);
        saxpy_cpu(l.kernel_weights_delta, l.workspace, l.inputs*l.outputs, 1./num, l.kernel_weights_delta);
        if (l.bias) saxpy_cpu(l.bias_delta, delta_n, l.outputs, 1./num, l.bias_delta);
    }
}

void update_connect_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        gemm(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.kernel_weights);
        saxpy_cpu(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs, rate, l.update_kernel_weights);
        if (l.bias){
            saxpy_cpu(l.update_bias_weights, delta_n, l.outputs, rate, l.update_bias_weights);
        }
    }
}

void connect_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize, float *n_delta)
{
    float *momentum_kernel_v;
    float *momentum_bias_v;
    if (decay != 0){
        saxpy_cpu(l.kernel_weights_delta, l.update_kernel_weights, l.inputs*l.outputs, 1-decay, l.workspace);
    }
    if (momentum != 0){
        multy_cpu(l.momentum_kernel_v, l.inputs*l.outputs, momentum, 1);
        saxpy_cpu(l.momentum_kernel_v, l.workspace, l.inputs*l.outputs, 1-dampening, l.momentum_kernel_v);
        if (nesterov){
            saxpy_cpu(l.workspace, l.momentum_kernel_v, l.inputs*l.outputs, momentum, l.workspace);
            momentum_kernel_v = l.workspace;
        } else {
            momentum_kernel_v = l.momentum_kernel_v;
        }
    }
    if (l.bias){
        if (decay != 0){
            saxpy_cpu(l.bias_delta, l.update_bias_weights, l.outputs, 1-decay, l.workspace);
        }
        if (momentum != 0){
            multy_cpu(l.momentum_bias_v, l.outputs, momentum, 1);
            saxpy_cpu(l.momentum_bias_v, l.workspace, l.outputs, 1-dampening, l.momentum_bias_v);
            if (nesterov){
                saxpy_cpu(l.workspace, l.momentum_bias_v, l.outputs, momentum, l.workspace);
                momentum_bias_v = l.workspace;
            } else {
                momentum_bias_v = l.momentum_bias_v;
            }
        }
    }
    if (maximize){
        saxpy_cpu(l.update_kernel_weights, momentum_kernel_v, l.inputs*l.outputs, -rate, l.update_kernel_weights);
        if (l.bias) saxpy_cpu(l.update_bias_weights, momentum_bias_v, l.outputs, -rate, l.update_bias_weights);
    } else {
        saxpy_cpu(l.update_kernel_weights, momentum_kernel_v, l.inputs*l.outputs, rate, l.update_kernel_weights);
        if (l.bias) saxpy_cpu(l.update_bias_weights, momentum_bias_v, l.outputs, rate, l.update_bias_weights);
    }
}

void connect_layer_AdamOptimizer(Layer l, float rate, float beta1, float beta2, float decay, int amsgrad, int maximize, float *n_delta)
{
    l.step_t += 1;
    float correction1 = 1 - pow(beta1, l.step_t);
    float correction2 = 1 - pow(beta2, l.step_t);
    float step_size = rate / correction1;
    float correction2_sqrt = sqrt(correction2);
    if (decay != 0){
        saxpy_cpu(l.kernel_weights_delta, l.update_kernel_weights, l.inputs*l.outputs, 1-decay, l.workspace);
    }
    lerp_cpu(l.exp_avg_kernel, l.workspace, l.inputs*l.outputs, beta1, l.exp_avg_kernel);
    lerp2_cpu(l.exp_avg_sq_kernel, l.workspace, l.inputs*l.outputs, beta2, l.exp_avg_sq_kernel);
    if (amsgrad){
        maximum_cpu(l.exp_avg_sq_kernel_max, l.exp_avg_sq_kernel, l.inputs*l.outputs, l.exp_avg_sq_kernel_max);
        denom_cpu(l.exp_avg_sq_kernel_max, correction2_sqrt, 0.00001, l.inputs*l.outputs, l.workspace);
    } else {
        denom_cpu(l.exp_avg_sq_kernel, correction2_sqrt, 0.00001, l.inputs*l.outputs, l.workspace);
    }
    addcdiv_cpu(l.update_kernel_weights, l.exp_avg_kernel, l.workspace, step_size, l.inputs*l.outputs, l.update_kernel_weights);
    if (l.bias){
        if (decay != 0){
            saxpy_cpu(l.bias_delta, l.update_bias_weights, l.outputs, 1-decay, l.workspace);
        }
        lerp_cpu(l.exp_avg_bias, l.workspace, l.outputs, beta1, l.exp_avg_bias);
        lerp2_cpu(l.exp_avg_sq_bias, l.workspace, l.outputs, beta2, l.exp_avg_sq_bias);
        if (amsgrad){
            maximum_cpu(l.exp_avg_sq_bias_max, l.exp_avg_sq_bias, l.outputs, l.exp_avg_sq_bias_max);
            denom_cpu(l.exp_avg_sq_bias_max, correction2_sqrt, 0.00001, l.outputs, l.workspace);
        } else {
            denom_cpu(l.exp_avg_sq_bias, correction2_sqrt, 0.00001, l.outputs, l.workspace);
        }
        addcdiv_cpu(l.update_bias_weights, l.exp_avg_bias, l.workspace, step_size, l.outputs, l.update_bias_weights);
    }
}

void refresh_connect_layer_weights(Layer l)
{
    memcpy(l.kernel_weights, l.update_kernel_weights, l.inputs*l.outputs*sizeof(float));
    if (l.bias){
        memcpy(l.bias_weights, l.update_bias_weights, l.outputs*sizeof(float));
    }
}

void save_connect_layer_weights(Layer l, FILE *fp)
{
    fwrite(l.kernel_weights, sizeof(float), l.inputs*l.outputs, fp);
    if (l.bias){
        fwrite(l.bias_weights, sizeof(float), l.outputs, fp);
    }
}

void zerograd_connect_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
    fill_cpu(l.kernel_weights_delta, l.inputs*l.outputs, 0, 1);
    fill_cpu(l.bias_delta, l.outputs, 0, 1);
}

void connect_constant_kernel_init(Layer l, float x)
{
    fill_cpu(l.kernel_weights, l.inputs*l.outputs, x, 1);
    memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
}

void connect_normal_kernel_init(Layer l, float mean, float std)
{
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        l.kernel_weights[i] = generate_normal(mean, std);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
}

void connect_uniform_kernel_init(Layer l, float min, float max)
{
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        l.kernel_weights[i] = rand_uniform(min, max);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
}

void connect_xavier_normal_kernel_init(Layer l, float gain)
{
    float fan_in = l.inputs;
    float fan_out = l.outputs;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        l.kernel_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
}

void connect_xavier_uniform_kernel_init(Layer l, float gain)
{
    float fan_in = l.inputs;
    float fan_out = l.outputs;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    float a = sqrt(3.0) * std;
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        l.kernel_weights[i] = rand_uniform(-a, a);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
}

void connect_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity)
{
    float fan = 0;
    float std = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.inputs;
    else if (0 == strcmp(mode, "fan_out")) fan = l.outputs;
    if (0 == strcmp(nonlinearity, "sigmoid")) a= 1;
    else if (0 == strcmp(nonlinearity, "tanh")) a = 5.0/3;
    else if (0 == strcmp(nonlinearity, "relu")) a = sqrt(2.0);
    else if (0 == strcmp(nonlinearity, "leaky_relu")){
        if (a == 0) a = 0.01;
        a = sqrt(2.0 / (1 + a*a));
    }
    else if (0 == strcmp(nonlinearity, "selu")) a = 3.0 / 4;
    std = a / sqrt(fan);
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        l.kernel_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
}

void connect_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity)
{
    float fan = 0;
    float std = 0;
    float bound = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.inputs;
    else if (0 == strcmp(mode, "fan_out")) fan = l.outputs;
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
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        l.kernel_weights[i] = rand_uniform(-bound, bound);
    }
    memcpy(l.update_kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float));
}

void connect_constant_bias_init(Layer l, float x)
{
    fill_cpu(l.bias_weights, l.outputs, x, 1);
    memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
}

void connect_normal_bias_init(Layer l, float mean, float std)
{
    for (int i = 0; i < l.outputs; ++i){
        l.bias_weights[i] = generate_normal(mean, std);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
}

void connect_uniform_bias_init(Layer l, float min, float max)
{
    for (int i = 0; i < l.outputs; ++i){
        l.bias_weights[i] = rand_uniform(min, max);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
}

void connect_xavier_normal_bias_init(Layer l, float gain)
{
    float fan_in = l.inputs;
    float fan_out = l.outputs;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.outputs; ++i){
        l.bias_weights[i] = rand_normal()*std;
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
}

void connect_xavier_uniform_bias_init(Layer l, float gain)
{
    float fan_in = l.inputs;
    float fan_out = l.outputs;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.outputs; ++i){
        l.bias_weights[i] = rand_uniform(-std, std);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
}

void connect_kaiming_normal_bias_init(Layer l, char *mode)
{
    float fan = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.inputs;
    else if (0 == strcmp(mode, "fan_out")) fan = l.outputs;
    else fan = l.inputs;
    float std = 1 / sqrt(fan);
    for (int i = 0; i < l.outputs; ++i){
        l.bias_weights[i] = generate_normal(0, std);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
}

void connect_kaiming_uniform_bias_init(Layer l, char *mode)
{
    float fan = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.inputs;
    else if (0 == strcmp(mode, "fan_out")) fan = l.outputs;
    else fan = l.inputs;
    float bound = 1 / sqrt(fan);
    for (int i = 0; i < l.outputs; ++i){
        l.bias_weights[i] = rand_uniform(-bound, bound);
    }
    memcpy(l.update_bias_weights, l.bias_weights, l.outputs*sizeof(float));
}
