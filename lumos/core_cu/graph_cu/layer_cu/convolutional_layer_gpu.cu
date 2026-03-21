#include "convolutional_layer_gpu.h"

void init_convolutional_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = (l->input_h + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_w = (l->input_w + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_c = l->filters;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = l->ksize * l->ksize * l->input_c * l->output_h * l->output_w + l->filters * l->ksize * l->ksize * l->input_c;

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->filters*l->ksize*l->ksize*l->input_c*sizeof(float));
    cudaMalloc((void**)&l->update_kernel_weights, l->filters*l->ksize*l->ksize*l->input_c*sizeof(float));
    if (l->bias){
        cudaMalloc((void**)&l->bias_weights, l->filters*sizeof(float));
        cudaMalloc((void**)&l->update_bias_weights, l->filters*sizeof(float));
        if (l->optimizer == SGD){
            cudaMalloc((void**)&l->momentum_bias_v, l->filters*sizeof(float));
            fill_gpu(l->momentum_bias_v, l->filters, 0, 1);
        }
    }
    if (l->optimizer == SGD){
        cudaMalloc((void**)&l->momentum_kernel_v, l->filters*l->ksize*l->ksize*l->input_c*sizeof(float));
        fill_gpu(l->momentum_kernel_v, l->filters*l->ksize*l->ksize*l->input_c, 0, 1);
    }

    fprintf(stderr, "Convolutional   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_convolutional_layer_gpu(Layer l, FILE *fp)
{
    if (fp){
        float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
        fread(kernel_weights, sizeof(float), l.ksize*l.ksize*l.filters*l.input_c, fp);
        cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
        free(kernel_weights);
        if (l.bias){
            float *bias_weights = (float*)calloc(l.filters, sizeof(float));
            fread(bias_weights, sizeof(float), l.filters, fp);
            cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            free(bias_weights);
        }
        return;
    }
    char *def_mode = (char *)"fan_in";
    char *def_nonlinearity = (char *)"leaky_relu";
    InitCptKernel initcptkernel = *l.initcptkernel;
    if (initcptkernel.initype == CONSTANT_I) convolutional_constant_kernel_init_gpu(l, initcptkernel.x);
    else if (initcptkernel.initype == NORMAL_I) convolutional_normal_kernel_init_gpu(l, initcptkernel.mean, initcptkernel.std);
    else if (initcptkernel.initype == XAVIER_NORMAL_I) convolutional_xavier_normal_kernel_init_gpu(l, initcptkernel.a);
    else if (initcptkernel.initype == XAVIER_UNIFORM_I) convolutional_xavier_uniform_kernel_init_gpu(l, initcptkernel.a);
    else if (initcptkernel.initype == KAIMING_NORMAL_I) convolutional_kaiming_normal_kernel_init_gpu(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
    else if (initcptkernel.initype == KAIMING_UNIFORM_I) convolutional_kaiming_uniform_kernel_init_gpu(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
    else convolutional_kaiming_uniform_kernel_init_gpu(l, sqrt(5.0), def_mode, def_nonlinearity);
    if (l.bias){
        InitCptBias initcptbias = *l.initcptbias;
        if (initcptbias.initype == CONSTANT_I) convolutional_constant_bias_init_gpu(l, initcptbias.x);
        else if (initcptbias.initype == NORMAL_I) convolutional_normal_bias_init_gpu(l, initcptbias.mean, initcptbias.std);
        else if (initcptbias.initype == UNIFORM_I) convolutional_uniform_bias_init_gpu(l, initcptbias.min, initcptbias.max);
        else if (initcptbias.initype == XAVIER_NORMAL_I) convolutional_xavier_normal_bias_init_gpu(l, initcptbias.a);
        else if (initcptbias.initype == XAVIER_UNIFORM_I) convolutional_xavier_uniform_bias_init_gpu(l, initcptbias.a);
        else if (initcptbias.initype == KAIMING_NORMAL_I) convolutional_kaiming_normal_bias_init_gpu(l, initcptbias.mode);
        else if (initcptbias.initype == KAIMING_UNIFORM_I) convolutional_kaiming_uniform_bias_init_gpu(l, initcptbias.mode);
        else convolutional_kaiming_uniform_bias_init_gpu(l, def_mode);
    }
}

void forward_convolutional_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm_gpu(0, 0, l.filters, l.ksize * l.ksize * l.input_c, l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             l.kernel_weights, l.workspace, output);
        if (l.bias){
            add_bias_gpu(output, l.bias_weights, l.filters, l.output_h * l.output_w);
        }
        activate_list_gpu(output, l.outputs, l.active);
    }
}

void backward_convolutional_layer_gpu(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        gradient_list_gpu(output, l.outputs, l.active);
        matrix_multiply_gpu(delta_n, output, l.outputs, delta_n);
        gemm_gpu(1, 0, l.filters, l.ksize * l.ksize * l.input_c,
             l.filters, l.output_h * l.output_w, 1,
             l.kernel_weights, delta_n, l.workspace);
        col2im_gpu(l.workspace, l.ksize, l.stride, l.pad, l.input_h, l.input_w, l.input_c, delta_l);
    }
}

void update_convolutional_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm_gpu(0, 1, l.filters, l.output_h * l.output_w,
             l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             delta_n, l.workspace, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w);
        saxpy_gpu(l.update_kernel_weights, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w, l.filters * l.ksize * l.ksize * l.input_c, rate, l.update_kernel_weights);
        if (l.bias){
            sum_channel_gpu(delta_n, l.output_h, l.output_w, l.output_c, rate, l.workspace);
            add_bias_gpu(l.update_bias_weights, l.workspace, l.output_c, 1);
        }
    }
}

void convolutional_layer_SGDOptimizer_gpu(Layer l, float rate, float momentum, float decay, int nesterov, int maximize, int num, float *n_delta)
{
    multy_gpu(l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c, 1-decay, 1);
    multy_gpu(l.update_bias_weights, l.output_c, 1-decay, 1);
    if (nesterov){
        saxpy_gpu(l.update_kernel_weights, l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c, momentum, l.update_kernel_weights);
    }
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_n = n_delta + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.workspace);
        gemm_gpu(0, 1, l.filters, l.output_h * l.output_w,
             l.ksize * l.ksize * l.input_c, l.output_h * l.output_w, 1,
             delta_n, l.workspace, l.workspace + l.ksize * l.ksize * l.input_c * l.output_h * l.output_w);
        saxpy_gpu(l.workspace+l.ksize*l.ksize*l.input_c*l.output_h*l.output_w, l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c, momentum, l.momentum_kernel_v);
        if (l.bias){
            sum_channel_gpu(delta_n, l.output_h, l.output_w, l.output_c, 1, l.workspace);
            saxpy_gpu(l.workspace, l.momentum_bias_v, l.output_c, momentum, l.momentum_bias_v);
        }
    }
    if (maximize){
        saxpy_gpu(l.update_kernel_weights, l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c, -rate, l.update_kernel_weights);
        saxpy_gpu(l.update_bias_weights, l.momentum_bias_v, l.outputs, -rate, l.update_bias_weights);
    } else {
        saxpy_gpu(l.update_kernel_weights, l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c, rate, l.update_kernel_weights);
        saxpy_gpu(l.update_bias_weights, l.momentum_bias_v, l.outputs, rate, l.update_bias_weights);
    }
}

void refresh_convolutional_layer_weights_gpu(Layer l)
{
    cudaMemcpy(l.kernel_weights, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyDeviceToDevice);
    if (l.bias){
        cudaMemcpy(l.bias_weights, l.update_bias_weights, l.filters*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void save_convolutional_layer_weights_gpu(Layer l, FILE *fp)
{
    float *kernel_weights = (float*)calloc(l.ksize*l.ksize*l.filters*l.input_c, sizeof(float));
    cudaMemcpy(kernel_weights, l.kernel_weights, l.ksize*l.ksize*l.filters*l.input_c*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(kernel_weights, sizeof(float), l.ksize*l.ksize*l.filters*l.input_c, fp);
    free(kernel_weights);
    if (l.bias){
        float *bias_weights = (float*)calloc(l.filters, sizeof(float));
        cudaMemcpy(bias_weights, l.bias_weights, l.filters*sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(bias_weights, sizeof(float), l.filters, fp);
        free(bias_weights);
    }
}

void zerograd_convolutional_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
}

void convolutional_constant_kernel_init_gpu(Layer l, float x)
{
    fill_gpu(l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c, x, 1);
    cudaMemcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
}

void convolutional_normal_kernel_init_gpu(Layer l, float mean, float std)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        kernel_weights[i] = generate_normal(mean, std);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void convolutional_uniform_kernel_init_gpu(Layer l, float min, float max)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        kernel_weights[i] = rand_uniform(min, max);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void convolutional_xavier_normal_kernel_init_gpu(Layer l, float gain)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        kernel_weights[i] = rand_normal()*std;
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void convolutional_xavier_uniform_kernel_init_gpu(Layer l, float gain)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    float a = sqrt(3.0) * std;
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c; ++i){
        l.kernel_weights[i] = rand_uniform(-a, a);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void convolutional_kaiming_normal_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
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
        kernel_weights[i] = rand_normal()*std;
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void convolutional_kaiming_uniform_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    float fan = 0;
    float std = 0;
    float bound = 0;
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c, sizeof(float));
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
        kernel_weights[i] = rand_uniform(-bound, bound);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void convolutional_constant_bias_init_gpu(Layer l, float x)
{
    fill_gpu(l.bias_weights, l.filters, x, 1);
    fill_gpu(l.update_bias_weights, l.filters, x, 1);
}

void convolutional_normal_bias_init_gpu(Layer l, float mean, float std)
{
    float *bias_weights = (float*)calloc(l.filters, sizeof(float));
    for (int i = 0; i < l.filters; ++i){
        bias_weights[i] = generate_normal(mean, std);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void convolutional_uniform_bias_init_gpu(Layer l, float min, float max)
{
    float *bias_weights = (float*)calloc(l.filters, sizeof(float));
    for (int i = 0; i < l.filters; ++i){
        bias_weights[i] = rand_uniform(min, max);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void convolutional_xavier_normal_bias_init_gpu(Layer l, float gain)
{
    float *bias_weights = (float*)calloc(l.filters, sizeof(float));
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters; ++i){
        bias_weights[i] = rand_normal()*std;
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void convolutional_xavier_uniform_bias_init_gpu(Layer l, float gain)
{
    float *bias_weights = (float*)calloc(l.filters, sizeof(float));
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters; ++i){
        bias_weights[i] = rand_uniform(-std, std);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void convolutional_kaiming_normal_bias_init_gpu(Layer l, char *mode)
{
    float *bias_weights = (float*)calloc(l.filters, sizeof(float));
    float fan = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) fan = l.ksize*l.ksize*l.filters;
    else fan = l.ksize*l.ksize*l.input_c;
    float std = 1 / sqrt(fan);
    for (int i = 0; i < l.filters; ++i){
        bias_weights[i] = generate_normal(0, std);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}

void convolutional_kaiming_uniform_bias_init_gpu(Layer l, char *mode)
{
    float *bias_weights = (float*)calloc(l.filters, sizeof(float));
    float fan = 0;
    if (0 == strcmp(mode, "fan_in")) fan = l.ksize*l.ksize*l.input_c;
    else if (0 == strcmp(mode, "fan_out")) fan = l.ksize*l.ksize*l.filters;
    else fan = l.ksize*l.ksize*l.input_c;
    float bound = 1 / sqrt(fan);
    for (int i = 0; i < l.filters; ++i){
        bias_weights[i] = rand_uniform(-bound, bound);
    }
    cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
    free(bias_weights);
}
