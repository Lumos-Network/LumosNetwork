#ifndef LOCAL_LAYER_GPU_H
#define LOCAL_LAYER_GPU_H

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
extern "C"{
#endif

Layer *make_local_layer_gpu(int filters, int ksize, int stride, int pad, int dilation, int bias, char *active);
void init_local_layer_gpu(Layer *l, int w, int h, int c, int subdivision);
void weightinit_local_layer_gpu(Layer l, FILE *fp);

void forward_local_layer_gpu(Layer l, int num);
void backward_local_layer_gpu(Layer l, int num, float *n_delta);
void refresh_local_layer_weights_gpu(Layer l);

void save_local_layer_weights_gpu(Layer l, FILE *fp);
void zerograd_local_layer_gpu(Layer l, int subdivision);

void local_constant_kernel_init_gpu(Layer l, float x);
void local_normal_kernel_init_gpu(Layer l, float mean, float std);
void local_uniform_kernel_init_gpu(Layer l, float min, float max);
void local_xavier_normal_kernel_init_gpu(Layer l, float gain);
void local_xavier_uniform_kernel_init_gpu(Layer l, float gain);
void local_kaiming_normal_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity);
void local_kaiming_uniform_kernel_init_gpu(Layer l, float a, char *mode, char *nonlinearity);

void local_layer_SGDOptimizer_gpu(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize);

#ifdef __cplusplus
}
#endif

#endif