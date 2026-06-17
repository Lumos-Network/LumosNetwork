#ifndef LOCAL_LAYER_H
#define LOCAL_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "image.h"
#include "active.h"
#include "bias.h"
#include "gemm.h"
#include "cpu.h"

#include "local_layer_gpu.h"

#ifdef __cplusplus
extern "C"{
#endif

Layer *make_local_layer(int filters, int ksize, int stride, int pad, int dilation, int bias, char *active);
void init_local_layer(Layer *l, int w, int h, int c, int subdivision);
void weightinit_local_layer(Layer l, FILE *fp);

void forward_local_layer(Layer l, int num);
void backward_local_layer(Layer l, int num, float *n_delta);
void refresh_local_layer_weights(Layer l);

void save_local_layer_weights(Layer l, FILE *fp);
void zerograd_local_layer(Layer l, int subdivision);

void local_constant_kernel_init(Layer l, float x);
void local_normal_kernel_init(Layer l, float mean, float std);
void local_uniform_kernel_init(Layer l, float min, float max);
void local_xavier_normal_kernel_init(Layer l, float gain);
void local_xavier_uniform_kernel_init(Layer l, float gain);
void local_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity);
void local_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity);

void local_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize);

#ifdef __cplusplus
}
#endif

#endif