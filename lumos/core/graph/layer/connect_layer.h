#ifndef CONNECT_LAYER_H
#define CONNECT_LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "bias.h"
#include "active.h"
#include "gemm.h"
#include "cpu.h"

#include "connect_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void denom_cpu(float *data, float x, float eps, int num, float *space);

Layer *make_connect_layer(int output, int bias, char *active);
void init_connect_layer(Layer *l, int w, int h, int c, int subdivision);
void weightinit_connect_layer(Layer l, FILE *fp);

void forward_connect_layer(Layer l, int num);
void backward_connect_layer(Layer l, int num, float *n_delta);
void update_connect_layer(Layer l, float rate, int num, float *n_delta);
void refresh_connect_layer_weights(Layer l);

void connect_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize, int num, float *n_delta);
void connect_layer_AdamOptimizer(Layer l, float rate, float beta1, float beta2, float decay, int amsgrad, int maximize, int num, float *n_delta);

void save_connect_layer_weights(Layer l, FILE *fp);
void zerograd_connect_layer(Layer l, int subdivision);

void connect_constant_kernel_init(Layer l, float x);
void connect_normal_kernel_init(Layer l, float mean, float std);
void connect_uniform_kernel_init(Layer l, float min, float max);
void connect_xavier_normal_kernel_init(Layer l, float gain);
void connect_xavier_uniform_kernel_init(Layer l, float gain);
void connect_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity);
void connect_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity);

void connect_constant_bias_init(Layer l, float x);
void connect_normal_bias_init(Layer l, float mean, float std);
void connect_uniform_bias_init(Layer l, float min, float max);
void connect_xavier_normal_bias_init(Layer l, float gain);
void connect_xavier_uniform_bias_init(Layer l, float gain);
void connect_kaiming_normal_bias_init(Layer l, char *mode);
void connect_kaiming_uniform_bias_init(Layer l, char *mode);

#ifdef __cplusplus
}
#endif

#endif