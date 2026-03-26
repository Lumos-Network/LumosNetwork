#include "connect_layer_gpu.h"

void init_connect_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
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

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->inputs*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->update_kernel_weights, l->inputs*l->outputs*sizeof(float));
    if (l->bias){
        cudaMalloc((void**)&l->bias_weights, l->outputs*sizeof(float));
        cudaMalloc((void**)&l->update_bias_weights, l->outputs*sizeof(float));
        if (l->optimizer == SGD){
            cudaMalloc((void**)&l->momentum_bias_v, l->outputs*sizeof(float));
            fill_gpu(l->momentum_bias_v, l->outputs, 0, 1);
        }
    }
    if (l->optimizer == SGD){
        cudaMalloc((void**)&l->momentum_kernel_v, l->inputs*l->outputs*sizeof(float));
        fill_gpu(l->momentum_kernel_v, l->inputs*l->outputs, 0, 1);
    }

    fprintf(stderr, "Connect         Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_connect_layer_gpu(Layer l, FILE *fp)
{
    if (fp){
        float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
        fread(kernel_weights, sizeof(float), l.outputs*l.inputs, fp);
        cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
        free(kernel_weights);
        if (l.bias){
            float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
            fread(bias_weights, sizeof(float), l.outputs, fp);
            cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
            free(bias_weights);
        }
        return;
    }
    char *def_mode = (char *)"fan_in";
    char *def_nonlinearity = (char *)"leaky_relu";
    InitCptKernel initcptkernel = *l.initcptkernel;
    if (initcptkernel.initype == CONSTANT_I) connect_constant_kernel_init_gpu(l, initcptkernel.x);
    else if (initcptkernel.initype == NORMAL_I) connect_normal_kernel_init_gpu(l, initcptkernel.mean, initcptkernel.std);
    else if (initcptkernel.initype == UNIFORM_I) connect_uniform_kernel_init_gpu(l, initcptkernel.min, initcptkernel.max);
    else if (initcptkernel.initype == XAVIER_NORMAL_I) connect_xavier_normal_kernel_init_gpu(l, initcptkernel.a);
    else if (initcptkernel.initype == XAVIER_UNIFORM_I) connect_xavier_uniform_kernel_init_gpu(l, initcptkernel.a);
    else if (initcptkernel.initype == KAIMING_NORMAL_I) connect_kaiming_normal_kernel_init_gpu(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
    else if (initcptkernel.initype == KAIMING_UNIFORM_I) connect_kaiming_uniform_kernel_init_gpu(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
    else connect_kaiming_uniform_kernel_init_gpu(l, sqrt(5.0), def_mode, def_nonlinearity);
    if (l.bias){
        InitCptBias initcptbias = *l.initcptbias;
        if (initcptbias.initype == CONSTANT_I) connect_constant_bias_init_gpu(l, initcptbias.x);
        else if (initcptbias.initype == NORMAL_I) connect_normal_bias_init_gpu(l, initcptbias.mean, initcptbias.std);
        else if (initcptbias.initype == UNIFORM_I) connect_uniform_bias_init_gpu(l, initcptbias.min, initcptbias.max);
        else if (initcptbias.initype == XAVIER_NORMAL_I) connect_xavier_normal_bias_init_gpu(l, initcptbias.a);
        else if (initcptbias.initype == XAVIER_UNIFORM_I) connect_xavier_uniform_bias_init_gpu(l, initcptbias.a);
        else if (initcptbias.initype == KAIMING_NORMAL_I) connect_kaiming_normal_bias_init_gpu(l, initcptbias.mode);
        else if (initcptbias.initype == KAIMING_UNIFORM_I) connect_kaiming_uniform_bias_init_gpu(l, initcptbias.mode);
        else connect_kaiming_uniform_bias_init_gpu(l, def_mode);
    }
}

void forward_connect_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        gemm_gpu(0, 0, l.outputs, l.inputs, l.inputs, 1,
             1, l.kernel_weights, input, output);
        if (l.bias){
            add_bias_gpu(output, l.bias_weights, l.ksize, 1);
        }
    }
    activate_list_gpu(l.output, num*l.outputs, l.output, l.active);
}

void backward_connect_layer_gpu(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list_gpu(output, l.outputs, l.workspace, l.active);
        matrix_multiply_gpu(delta_n, l.workspace, l.outputs, delta_n);
        gemm_gpu(1, 0, l.output_c, l.input_c, l.output_c, l.input_w, 1,
             l.kernel_weights, delta_n, delta_l);
        gemm_gpu(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.workspace);
        saxpy_gpu(l.kernel_weights_delta, l.workspace, l.inputs*l.outputs, 1./num, l.kernel_weights_delta);
        if (l.bias) saxpy_gpu(l.bias_delta, delta_n, l.outputs, 1./num, l.bias_delta);
    }
}

void update_connect_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        gemm_gpu(0, 1, l.output_c, l.output_w,
             l.input_c, l.input_w, 1,
             delta_n, input, l.workspace);
        saxpy_gpu(l.update_kernel_weights, l.workspace, l.output_c * l.input_c, rate, l.update_kernel_weights);
        if (l.bias){
            saxpy_gpu(l.update_bias_weights, delta_n, l.outputs, rate, l.update_bias_weights);
        }
    }
}

void connect_layer_SGDOptimizer_gpu(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize, float *n_delta)
{
    float *momentum_kernel_v;
    float *momentum_bias_v;
    if (decay != 0){
        saxpy_gpu(l.kernel_weights_delta, l.update_kernel_weights, l.inputs*l.outputs, 1-decay, l.workspace);
    }
    if (momentum != 0){
        multy_gpu(l.momentum_kernel_v, l.inputs*l.outputs, momentum, 1);
        saxpy_gpu(l.momentum_kernel_v, l.workspace, l.inputs*l.outputs, 1-dampening, l.momentum_kernel_v);
        if (nesterov){
            saxpy_gpu(l.workspace, l.momentum_kernel_v, l.inputs*l.outputs, momentum, l.workspace);
            momentum_kernel_v = l.workspace;
        } else {
            momentum_kernel_v = l.momentum_kernel_v;
        }
    }
    if (l.bias){
        if (decay != 0){
            saxpy_gpu(l.bias_delta, l.update_bias_weights, l.outputs, 1-decay, l.workspace);
        }
        if (momentum != 0){
            multy_gpu(l.momentum_bias_v, l.outputs, momentum, 1);
            saxpy_gpu(l.momentum_bias_v, l.workspace, l.outputs, 1-dampening, l.momentum_bias_v);
            if (nesterov){
                saxpy_gpu(l.workspace, l.momentum_bias_v, l.outputs, momentum, l.workspace);
                momentum_bias_v = l.workspace;
            } else {
                momentum_bias_v = l.momentum_bias_v;
            }
        }
    }
    if (maximize){
        saxpy_gpu(l.update_kernel_weights, momentum_kernel_v, l.inputs*l.outputs, -rate, l.update_kernel_weights);
        if (l.bias) saxpy_gpu(l.update_bias_weights, momentum_bias_v, l.outputs, -rate, l.update_bias_weights);
    } else {
        saxpy_gpu(l.update_kernel_weights, momentum_kernel_v, l.inputs*l.outputs, rate, l.update_kernel_weights);
        if (l.bias) saxpy_gpu(l.update_bias_weights, momentum_bias_v, l.outputs, rate, l.update_bias_weights);
    }
}

void refresh_connect_layer_weights_gpu(Layer l)
{
    cudaMemcpy(l.kernel_weights, l.update_kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
    if (l.bias){
        cudaMemcpy(l.bias_weights, l.update_bias_weights, l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void save_connect_layer_weights_gpu(Layer l, FILE *fp)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    cudaMemcpy(kernel_weights, l.kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(kernel_weights, sizeof(float), l.inputs*l.outputs, fp);
    free(kernel_weights);
    if (l.bias){
        float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
        cudaMemcpy(bias_weights, l.bias_weights, l.outputs*sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(bias_weights, sizeof(float), l.outputs, fp);
        free(bias_weights);
    }
}

void zerograd_connect_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
    fill_gpu(l.kernel_weights_delta, l.inputs*l.outputs, 0, 1);
    fill_gpu(l.bias_delta, l.outputs, 0, 1);
}

void connect_constant_kernel_init_gpu(Layer l, float x)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    fill_cpu(kernel_weights, l.inputs*l.outputs, x, 0);
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_normal_kernel_init_gpu(Layer l, float mean, float std)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = generate_normal(mean, std);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_uniform_kernel_init_gpu(Layer l, float min, float max)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = rand_uniform(min, max);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_xavier_normal_kernel_init_gpu(Layer l, float gain)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    float fan_in = l.inputs;
    float fan_out = l.outputs;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = rand_normal()*std;
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_xavier_uniform_kernel_init_gpu(Layer l, float gain)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
    float fan_in = l.inputs;
    float fan_out = l.outputs;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    float a = sqrt(3.0) * std;
    for (int i = 0; i < l.inputs*l.outputs; ++i){
        kernel_weights[i] = rand_uniform(-a, a);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_kaiming_normal_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
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
        kernel_weights[i] = rand_normal()*std;
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_kaiming_uniform_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    float *kernel_weights = (float*)calloc(l.inputs*l.outputs, sizeof(float));
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
        kernel_weights[i] = rand_uniform(-bound, bound);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.inputs*l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void connect_constant_bias_init_gpu(Layer l, float x)
{
    float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
    fill_cpu(bias_weights, l.outputs, x, 1);
    cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void connect_normal_bias_init_gpu(Layer l, float mean, float std)
{
    float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
    for (int i = 0; i < l.outputs; ++i){
        bias_weights[i] = generate_normal(mean, std);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void connect_uniform_bias_init_gpu(Layer l, float min, float max)
{
    float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
    for (int i = 0; i < l.outputs; ++i){
        bias_weights[i] = rand_uniform(min, max);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void connect_xavier_normal_bias_init_gpu(Layer l, float gain)
{
    float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
    float fan_in = l.inputs;
    float fan_out = l.outputs;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.outputs; ++i){
        bias_weights[i] = rand_normal()*std;
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void connect_xavier_uniform_bias_init_gpu(Layer l, float gain)
{
    float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
    float fan_in = l.inputs;
    float fan_out = l.outputs;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.outputs; ++i){
        bias_weights[i] = rand_uniform(-std, std);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void connect_kaiming_normal_bias_init_gpu(Layer l, char *mode)
{
    float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
    float fan = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.inputs;
    else if (0 == strcmp(mode, "fan_out")) fan = l.outputs;
    else fan = l.inputs;
    float std = 1 / sqrt(fan);
    for (int i = 0; i < l.outputs; ++i){
        bias_weights[i] = generate_normal(0, std);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void connect_kaiming_uniform_bias_init_gpu(Layer l, char *mode)
{
    float *bias_weights = (float*)calloc(l.outputs, sizeof(float));
    float fan = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.inputs;
    else if (0 == strcmp(mode, "fan_out")) fan = l.outputs;
    else fan = l.inputs;
    float bound = 1 / sqrt(fan);
    for (int i = 0; i < l.outputs; ++i){
        bias_weights[i] = rand_uniform(-bound, bound);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.outputs*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}
