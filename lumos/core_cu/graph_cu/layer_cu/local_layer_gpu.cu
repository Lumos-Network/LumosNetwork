#include "local_layer_gpu.h"

void init_local_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
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

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w*sizeof(float));
    cudaMalloc((void**)&l->update_kernel_weights, l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights_delta, l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w*sizeof(float));
    if (l->optimizer == SGD){
        cudaMalloc((void**)&l->momentum_kernel_v, l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w*sizeof(float));
        fill_gpu(l->momentum_kernel_v, l->filters*l->ksize*l->ksize*l->input_c*l->output_h*l->output_w, 0, 1);
    }

    fprintf(stderr, "local   Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void weightinit_local_layer_gpu(Layer l, FILE *fp)
{
    if (fp){
        int flag = 0;
        int weights_num = l.filters*l.ksize*l.ksize*l.input_c*l.output_h*l.output_w;
        float *weights = (float*)malloc(weights_num*sizeof(float));
        flag = fread(weights, sizeof(float), weights_num, fp);
        if (flag == weights_num){
            cudaMemcpy(l.kernel_weights, weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_h*l.output_w*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.update_kernel_weights, weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_h*l.output_w*sizeof(float), cudaMemcpyHostToDevice);
            return;
        }
        free(weights);
    }
    char *def_mode = (char *)"fan_in";
    char *def_nonlinearity = (char *)"leaky";
    if (l.initcptkernel == NULL){
        local_kaiming_uniform_kernel_init_gpu(l, sqrt(5.0), def_mode, def_nonlinearity);
    } else {
        InitCptKernel initcptkernel = *l.initcptkernel;
        if (initcptkernel.initype == CONSTANT_I) local_constant_kernel_init_gpu(l, initcptkernel.x);
        else if (initcptkernel.initype == NORMAL_I) local_normal_kernel_init_gpu(l, initcptkernel.mean, initcptkernel.std);
        else if (initcptkernel.initype == XAVIER_NORMAL_I) local_xavier_normal_kernel_init_gpu(l, initcptkernel.a);
        else if (initcptkernel.initype == XAVIER_UNIFORM_I) local_xavier_uniform_kernel_init_gpu(l, initcptkernel.a);
        else if (initcptkernel.initype == KAIMING_NORMAL_I) local_kaiming_normal_kernel_init_gpu(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
        else if (initcptkernel.initype == KAIMING_UNIFORM_I) local_kaiming_uniform_kernel_init_gpu(l, initcptkernel.a, initcptkernel.mode, initcptkernel.nonlinearity);
        else local_kaiming_uniform_kernel_init_gpu(l, sqrt(5.0), def_mode, def_nonlinearity);
    }
}

void forward_local_layer_gpu(Layer l, int num)
{
    fill_gpu(l.output, num*l.outputs, 0, 1);
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.dilation, l.workspace);
        for (int j = 0; j < l.output_w*l.output_h; ++j){
            float *weight = l.kernel_weights + j*l.ksize*l.ksize*l.input_c*l.filters;
            float *data = l.workspace + j;
            float *out = output + j;
            gemm_gpu(0, 0, l.filters, 1, l.ksize*l.ksize*l.input_c, 1,
                weight, l.ksize*l.ksize*l.input_c, data, l.output_w*l.output_h,
                1, out, l.output_w*l.output_h);
        }
    }
    if (l.active == LINEAR) return;
    activate_list_gpu(l.output, num*l.outputs, l.output, l.active);
}

void backward_local_layer_gpu(Layer l, int num, float *n_delta)
{
    gradient_list_gpu(l.output, num*l.outputs, n_delta, l.active);
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        im2col_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, l.dilation, l.workspace);
        for (int j = 0; j < l.output_w*l.output_h; ++j){
            float *delta = delta_n + j;
            float *data = l.workspace + j;
            float *weight_delta = l.kernel_weights_delta + j*l.ksize*l.ksize*l.input_c*l.filters;
            gemm_gpu(0, 1, l.filters, l.ksize*l.ksize*l.input_c, 1, 1,
                delta, l.output_w*l.output_h, data, l.output_w*l.output_h, 1,
                weight_delta, l.ksize*l.ksize*l.input_c);
            float *weight = l.kernel_weights + j*l.ksize*l.ksize*l.input_c*l.filters;
            gemm_gpu(1, 0, l.ksize*l.ksize*l.input_c, 1, l.filters, 1,
                weight, l.ksize*l.ksize*l.input_c, delta, l.output_w*l.output_h, 0,
                data, l.output_w*l.output_h);
        }
        col2im_gpu(l.workspace, l.ksize, l.stride, l.pad, l.dilation, l.input_h, l.input_w, l.input_c, delta_l);
    }
}

void local_layer_SGDOptimizer_gpu(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize)
{
    float *momentum_kernel_v;
    if (decay != 0){
        saxpy_gpu(l.kernel_weights_delta, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, decay, l.kernel_weights_delta);
    }
    if (momentum != 0){
        multy_gpu(l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, momentum, 1);
        saxpy_gpu(l.momentum_kernel_v, l.kernel_weights_delta, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, 1-dampening, l.momentum_kernel_v);
        if (nesterov){
            saxpy_gpu(l.kernel_weights_delta, l.momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, momentum, l.kernel_weights_delta);
            momentum_kernel_v = l.kernel_weights_delta;
        } else {
            momentum_kernel_v = l.momentum_kernel_v;
        }
    }
    if (maximize){
        saxpy_gpu(l.update_kernel_weights, momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, -rate, l.update_kernel_weights);
    } else {
        saxpy_gpu(l.update_kernel_weights, momentum_kernel_v, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, rate, l.update_kernel_weights);
    }
}

void refresh_local_layer_weights_gpu(Layer l)
{
    cudaMemcpy(l.kernel_weights, l.update_kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyDeviceToDevice);
}

void save_local_layer_weights_gpu(Layer l, FILE *fp)
{
    float *kernel_weights = (float*)calloc(l.ksize*l.ksize*l.filters*l.input_c*l.output_w*l.output_h, sizeof(float));
    cudaMemcpy(kernel_weights, l.kernel_weights, l.ksize*l.ksize*l.filters*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(kernel_weights, sizeof(float), l.ksize*l.ksize*l.filters*l.input_c*l.output_w*l.output_h, fp);
    free(kernel_weights);
}

void zerograd_local_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
    fill_gpu(l.kernel_weights_delta, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, 0, 1);
}

void local_constant_kernel_init_gpu(Layer l, float x)
{
    fill_gpu(l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, x, 1);
    cudaMemcpy(l.update_kernel_weights, l.kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
}

void local_normal_kernel_init_gpu(Layer l, float mean, float std)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, sizeof(float));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        kernel_weights[i] = generate_normal(mean, std);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void local_uniform_kernel_init_gpu(Layer l, float min, float max)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, sizeof(float));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        kernel_weights[i] = rand_uniform(min, max);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void local_xavier_normal_kernel_init_gpu(Layer l, float gain)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, sizeof(float));
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        kernel_weights[i] = rand_normal()*std;
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void local_xavier_uniform_kernel_init_gpu(Layer l, float gain)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, sizeof(float));
    float fan_in = l.ksize*l.ksize*l.input_c;
    float fan_out = l.ksize*l.ksize*l.filters;
    float std = gain * sqrt(2.0 / (fan_in + fan_out));
    float a = sqrt(3.0) * std;
    for (int i = 0; i < l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h; ++i){
        l.kernel_weights[i] = rand_uniform(-a, a);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void local_kaiming_normal_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, sizeof(float));
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
        kernel_weights[i] = rand_normal()*std;
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}

void local_kaiming_uniform_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity)
{
    float fan = 0;
    float std = 0;
    float bound = 0;
    float *kernel_weights = (float*)calloc(l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h, sizeof(float));
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
        kernel_weights[i] = rand_uniform(-bound, bound);
    }
    cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*l.ksize*l.ksize*l.input_c*l.output_w*l.output_h*sizeof(float), cudaMemcpyHostToDevice);
    free(kernel_weights);
}