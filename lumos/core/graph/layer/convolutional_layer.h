#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "image.h"
#include "active.h"
#include "bias.h"
#include "gemm.h"
#include "cpu.h"

#include "convolutional_layer_gpu.h"

#ifdef __cplusplus
extern "C"{
#endif

Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, char *active);
void init_convolutional_layer(Layer *l, int w, int h, int c, int subdivision);
void weightinit_convolutional_layer(Layer l, FILE *fp);

void forward_convolutional_layer(Layer l, int num);
void backward_convolutional_layer(Layer l, int num, float *n_delta);
void update_convolutional_layer(Layer l, float rate, int num, float *n_delta);
void refresh_convolutional_layer_weights(Layer l);

void save_convolutional_layer_weights(Layer l, FILE *fp);
void zerograd_convolutional_layer(Layer l, int subdivision);

void convolutional_constant_kernel_init(Layer l, float x);
void convolutional_normal_kernel_init(Layer l, float mean, float std);
void convolutional_uniform_kernel_init(Layer l, float min, float max);
void convolutional_xavier_normal_kernel_init(Layer l, float gain);
void convolutional_xavier_uniform_kernel_init(Layer l, float gain);
void convolutional_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity);
void convolutional_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity);

void convolutional_constant_bias_init(Layer l, float x);
void convolutional_normal_bias_init(Layer l, float mean, float std);
void convolutional_uniform_bias_init(Layer l, float min, float max);
void convolutional_xavier_normal_bias_init(Layer l, float gain);
void convolutional_xavier_uniform_bias_init(Layer l, float gain);
void convolutional_kaiming_normal_bias_init(Layer l, char *mode);
void convolutional_kaiming_uniform_bias_init(Layer l, char *mode);

void convolutional_layer_SGDOptimizer(Layer l, float rate, float momentum, float decay, int nesterov, int maximize, int num, float *n_delta);

#ifdef __cplusplus
}
#endif

#endif