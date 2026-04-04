#include "normalization_layer_gpu.h"

void init_normalization_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
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

    l->workspace_size = subdivision*l->outputs;;

    cudaMalloc((void**)&l->mean, l->filters*sizeof(float));
    cudaMalloc((void**)&l->variance, l->filters*sizeof(float));
    cudaMalloc((void**)&l->rolling_mean, l->filters*sizeof(float));
    cudaMalloc((void**)&l->rolling_variance, l->filters*sizeof(float));
    cudaMalloc((void**)&l->mean_delta, l->filters*sizeof(float));
    cudaMalloc((void**)&l->variance_delta, l->filters*sizeof(float));
    cudaMalloc((void**)&l->kernel_weights, l->filters*sizeof(float));
    cudaMalloc((void**)&l->bias_weights, l->filters*sizeof(float));
    if (l->affine){
        cudaMalloc((void**)&l->update_kernel_weights, l->filters*sizeof(float));
        cudaMalloc((void**)&l->update_bias_weights, l->filters*sizeof(float));
        cudaMalloc((void**)&l->kernel_weights_delta, l->filters*sizeof(float));
        cudaMalloc((void**)&l->bias_delta, l->filters*sizeof(float));
        if (l->optimizer == SGD){
            cudaMalloc((void**)&l->momentum_kernel_v, l->filters*sizeof(float));
            fill_gpu(l->momentum_kernel_v, l->filters, 0, 1);
            cudaMalloc((void**)&l->momentum_bias_v, l->filters*sizeof(float));
            fill_gpu(l->momentum_bias_v, l->filters, 0, 1);
        }
    }
    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));
    cudaMalloc((void**)&l->norm_x, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "Normalization   Layer\n");
}

void weightinit_normalization_layer_gpu(Layer l, FILE *fp)
{
    if (fp){
        float *rolling_mean = (float*)calloc(l.filters, sizeof(float));
        float *rolling_variance = (float*)calloc(l.filters, sizeof(float));
        fread(rolling_mean, sizeof(float), l.filters, fp);
        fread(rolling_variance, sizeof(float), l.filters, fp);
        cudaMemcpy(l.rolling_mean, rolling_mean, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.rolling_variance, rolling_variance, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        if (l.affine){
            float *kernel_weights = (float*)calloc(l.filters, sizeof(float));
            float *bias_weights = (float*)calloc(l.filters, sizeof(float));
            fread(kernel_weights, sizeof(float), l.filters, fp);
            fread(bias_weights, sizeof(float), l.filters, fp);
            cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            free(kernel_weights);
            free(bias_weights);
        } else {
            fill_gpu(l.kernel_weights, l.filters, 1, 1);
            fill_gpu(l.bias_weights, l.filters, 0, 1);
        }
        free(rolling_mean);
        free(rolling_variance);
        return;
    }
    fill_gpu(l.kernel_weights, l.filters, 1, 1);
    fill_gpu(l.bias_weights, l.filters, 0, 1);
    if (l.affine){
        fill_gpu(l.update_kernel_weights, l.filters, 1, 1);
        fill_gpu(l.update_bias_weights, l.filters, 0, 1);
    }
}

void forward_normalization_layer_gpu(Layer l, int num)
{
    if (l.status){
        normalize_mean_gpu(l.input, l.ksize, l.filters, num, l.mean);
        normalize_variance_gpu(l.input, l.ksize, l.filters, num, l.mean, l.variance);
        multy_gpu(l.rolling_mean, l.filters, 1-l.bn_momentum, 1);
        multy_gpu(l.rolling_variance, l.filters, 1-l.bn_momentum, 1);
        saxpy_gpu(l.rolling_mean, l.mean, l.filters, l.bn_momentum, l.rolling_mean);
        saxpy_gpu(l.rolling_variance, l.variance, l.filters, l.bn_momentum, l.rolling_variance);
    }
    for (int i = 0; i < num; ++i){
        float *input = l.input + l.inputs*i;
        float *output = l.output + l.outputs*i;
        float *norm_x = l.norm_x + l.inputs*i;
        if (l.status) normalize_gpu(input, l.mean, l.variance, l.ksize, l.filters, output);
        if (!l.status) normalize_gpu(input, l.rolling_mean, l.rolling_variance, l.ksize, l.filters, output);
        cudaMemcpy(norm_x, output, l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
        scale_bias_gpu(output, l.kernel_weights, l.filters, l.ksize);
        add_bias_gpu(output, l.bias_weights, l.filters, l.ksize);
    }
    activate_list_gpu(l.output, num*l.outputs, l.output, l.active);
}

void backward_normalization_layer_gpu(Layer l, int num, float *n_delta)
{
    gradient_list_gpu(l.output, num*l.outputs, l.workspace, l.active);
    matrix_multiply_gpu(n_delta, l.workspace, num*l.outputs, n_delta);
    cudaMemcpy(l.delta, n_delta, num*l.inputs*sizeof(float), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < num; ++i){
        float *input = l.input + i*l.inputs;
        float *delta_l = l.delta + i*l.inputs;
        float *delta_n = n_delta + i*l.outputs;
        scale_bias_gpu(delta_l, l.kernel_weights, l.filters, l.ksize);
        gradient_normalize_mean_gpu(delta_l, l.variance, l.ksize, l.filters, l.mean_delta);
        gradient_normalize_variance_gpu(delta_l, input, l.mean, l.variance, l.ksize, l.filters, l.variance_delta);
        gradient_normalize_gpu(input, l.mean, l.variance, l.mean_delta, l.variance_delta, l.ksize, l.filters, delta_l, delta_l);
        if (l.affine){
            float *norm_x = l.norm_x + i*l.inputs;
            gradient_scale_gpu(norm_x, l.mean, l.variance, delta_n, l.ksize, l.filters, l.workspace);
            saxpy_gpu(l.kernel_weights_delta, l.workspace, l.filters, 1./num, l.kernel_weights_delta);
            gradient_bias_gpu(delta_n, l.ksize, l.filters, l.workspace);
            saxpy_gpu(l.bias_delta, l.workspace, l.filters, 1./num, l.bias_delta);
        }
    }
}

void normalization_layer_SGDOptimizer_gpu(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize)
{
    if (!l.affine) return;
    float *momentum_kernel_v;
    float *momentum_bias_v;
    if (decay != 0){
        saxpy_gpu(l.kernel_weights_delta, l.update_kernel_weights, l.filters, decay, l.kernel_weights_delta);
    }
    if (momentum != 0){
        multy_gpu(l.momentum_kernel_v, l.filters, momentum, 1);
        saxpy_gpu(l.momentum_kernel_v, l.kernel_weights_delta, l.filters, 1-dampening, l.momentum_kernel_v);
        if (nesterov){
            saxpy_gpu(l.kernel_weights_delta, l.momentum_kernel_v, l.filters, momentum, l.kernel_weights_delta);
            momentum_kernel_v = l.kernel_weights_delta;
        } else {
            momentum_kernel_v = l.momentum_kernel_v;
        }
    }
    if (decay != 0){
        saxpy_gpu(l.bias_delta, l.update_bias_weights, l.filters, decay, l.bias_delta);
    }
    if (momentum != 0){
        multy_gpu(l.momentum_bias_v, l.filters, momentum, 1);
        saxpy_gpu(l.momentum_bias_v, l.bias_delta, l.filters, 1-dampening, l.momentum_bias_v);
        if (nesterov){
            saxpy_gpu(l.bias_delta, l.momentum_bias_v, l.filters, momentum, l.bias_delta);
            momentum_bias_v = l.bias_delta;
        } else {
            momentum_bias_v = l.momentum_bias_v;
        }
    }
    if (maximize){
        saxpy_gpu(l.update_kernel_weights, momentum_kernel_v, l.filters, -rate, l.update_kernel_weights);
        saxpy_gpu(l.update_bias_weights, momentum_bias_v, l.filters, -rate, l.update_bias_weights);
    } else {
        saxpy_gpu(l.update_kernel_weights, momentum_kernel_v, l.filters, rate, l.update_kernel_weights);
        saxpy_gpu(l.update_bias_weights, momentum_bias_v, l.filters, rate, l.update_bias_weights);
    }
}

void refresh_normalization_layer_weights_gpu(Layer l)
{
    if (l.affine){
        cudaMemcpy(l.kernel_weights, l.update_kernel_weights, l.filters*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(l.bias_weights, l.update_bias_weights, l.filters*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void save_normalization_layer_weights_gpu(Layer l, FILE *fp)
{
    float *kernel_weights = (float*)calloc(l.filters, sizeof(float));
    float *bias_weights = (float*)calloc(l.filters, sizeof(float));
    float *rolling_mean = (float*)calloc(l.filters, sizeof(float));
    float *rolling_variance = (float*)calloc(l.filters, sizeof(float));
    cudaMemcpy(kernel_weights, l.kernel_weights, l.filters*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_weights, l.bias_weights, l.filters*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rolling_mean, l.rolling_mean, l.filters*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rolling_variance, l.rolling_variance, l.filters*sizeof(float), cudaMemcpyDeviceToHost);
    fwrite(kernel_weights, sizeof(float), l.filters, fp);
    fwrite(bias_weights, sizeof(float), l.filters, fp);
    fwrite(rolling_mean, sizeof(float), l.filters, fp);
    fwrite(rolling_variance, sizeof(float), l.filters, fp);
    free(kernel_weights);
    free(bias_weights);
    free(rolling_mean);
    free(rolling_variance);
}

void zerograd_normalization_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
    if (l.affine){
        fill_gpu(l.kernel_weights_delta, l.filters, 0, 1);
        fill_gpu(l.bias_delta, l.filters, 0, 1);
    }
}