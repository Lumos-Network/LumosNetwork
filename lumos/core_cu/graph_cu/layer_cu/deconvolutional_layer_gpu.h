#ifndef DEdeconvolutional_LAYER_GPU_H
#define DEdeconvolutional_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "random.h"
#include "cpu.h"
#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"
#include "active_gpu.h"
#include "gemm_gpu.h"
#include "im2col_gpu.h"
#include "bias_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_deconvolutional_layer_gpu(Layer *l, int w, int h, int c, int subdivision);
void weightinit_deconvolutional_layer_gpu(Layer l, FILE *fp);
void forward_deconvolutional_layer_gpu(Layer l, int num);
void backward_deconvolutional_layer_gpu(Layer l, int num, float *n_delta);
void refresh_deconvolutional_layer_weights_gpu(Layer l);
void save_deconvolutional_layer_weights_gpu(Layer l, FILE *fp);
void zerograd_deconvolutional_layer_gpu(Layer l, int subdivision);

void deconvolutional_constant_kernel_init_gpu(Layer l, float x);
void deconvolutional_normal_kernel_init_gpu(Layer l, float mean, float std);
void deconvolutional_uniform_kernel_init_gpu(Layer l, float min, float max);
void deconvolutional_xavier_normal_kernel_init_gpu(Layer l, float gain);
void deconvolutional_xavier_uniform_kernel_init_gpu(Layer l, float gain);
void deconvolutional_kaiming_normal_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity);
void deconvolutional_kaiming_uniform_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity);

void deconvolutional_constant_bias_init_gpu(Layer l, float x);
void deconvolutional_normal_bias_init_gpu(Layer l, float mean, float std);
void deconvolutional_uniform_bias_init_gpu(Layer l, float min, float max);
void deconvolutional_xavier_normal_bias_init_gpu(Layer l, float gain);
void deconvolutional_xavier_uniform_bias_init_gpu(Layer l, float gain);
void deconvolutional_kaiming_normal_bias_init_gpu(Layer l, char *mode);
void deconvolutional_kaiming_uniform_bias_init_gpu(Layer l, char *mode);

void deconvolutional_bilinearinterp_init_gpu(Layer l);

void deconvolutional_layer_SGDOptimizer_gpu(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize);

#ifdef __cplusplus
}
#endif
#endif
