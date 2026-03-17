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
    if (h == 1 && w == 1){
        l->filters = 1;
        l->ksize = l->inputs;
    }

    l->workspace_size = 0;

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
    }
    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "Normalization   Layer\n");
}

void weightinit_normalization_layer_gpu(Layer l, FILE *fp)
{
    if (fp){
        float *kernel_weights = (float*)calloc(l.filters, sizeof(float));
        float *bias_weights = (float*)calloc(l.filters, sizeof(float));
        float *rolling_mean = (float*)calloc(l.filters, sizeof(float));
        float *rolling_variance = (float*)calloc(l.filters, sizeof(float));
        fread(kernel_weights, sizeof(float), l.filters, fp);
        fread(bias_weights, sizeof(float), l.filters, fp);
        fread(rolling_mean, sizeof(float), l.filters, fp);
        fread(rolling_variance, sizeof(float), l.filters, fp);
        cudaMemcpy(l.kernel_weights, kernel_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.rolling_mean, rolling_mean, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(l.rolling_variance, rolling_variance, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        if (l.affine){
            cudaMemcpy(l.update_kernel_weights, kernel_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(l.update_bias_weights, bias_weights, l.filters*sizeof(float), cudaMemcpyHostToDevice);
        }
        free(kernel_weights);
        free(bias_weights);
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
        normalize_mean_gpu(l.output, l.ksize, l.filters, num, l.mean);
        normalize_variance_gpu(l.output, l.ksize, l.filters, num, l.mean, l.variance);
        multy_gpu(l.rolling_mean, l.filters, 1-l.bn_momentum, 1);
        multy_gpu(l.rolling_variance, l.filters, 1-l.bn_momentum, 1);
        saxpy_gpu(l.rolling_mean, l.mean, l.filters, l.bn_momentum, l.rolling_mean);
        saxpy_gpu(l.rolling_variance, l.variance, l.filters, l.bn_momentum, l.rolling_variance);
    }
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.output + offset_o;
        float *output = l.output + offset_o;
        if (l.status) normalize_gpu(input, l.mean, l.variance, l.ksize, l.filters, output);
        if (!l.status) normalize_gpu(input, l.rolling_mean, l.rolling_variance, l.ksize, l.filters, output);
        scale_bias_gpu(output, l.kernel_weights, l.filters, l.ksize);
        add_bias_gpu(output, l.bias_weights, l.filters, l.ksize);
    }
}

void backward_normalization_layer_gpu(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_o = i * l.outputs;
        float *input = l.input + offset_o;
        float *delta_l = l.delta + offset_o;
        float *delta_n = n_delta + offset_o;
        cudaMemcpy(delta_l, delta_n, l.outputs*sizeof(float), cudaMemcpyDeviceToDevice);
        scale_bias_gpu(delta_l, l.kernel_weights, l.filters, l.ksize);
        gradient_normalize_mean_gpu(delta_l, l.variance, l.ksize, l.filters, l.mean_delta);
        gradient_normalize_variance_gpu(delta_l, input, l.mean, l.variance, l.ksize, l.filters, l.variance_delta);
        gradient_normalize_gpu(input, l.mean, l.variance, l.mean_delta, l.variance_delta, l.ksize, l.filters, delta_l, delta_l);
    }
}

void update_normalization_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    if (!l.affine) return;
    for (int i = 0; i < num; ++i){
        float *delta_n = n_delta + i*l.outputs;
        float *input = l.input + i*l.inputs;
        update_scale_gpu(input, delta_n, l.ksize, l.filters, rate, l.update_kernel_weights);
        update_bias_gpu(delta_n, l.ksize, l.filters, rate, l.update_bias_weights);
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

void free_normalization_layer_gpu(Layer l)
{
    cudaFree(l.mean);
    cudaFree(l.variance);
    cudaFree(l.rolling_mean);
    cudaFree(l.rolling_variance);
    cudaFree(l.mean_delta);
    cudaFree(l.variance_delta);
    cudaFree(l.kernel_weights);
    cudaFree(l.bias_weights);
    if (l.affine){
        cudaFree(l.update_kernel_weights);
        cudaFree(l.update_bias_weights);
    }
}