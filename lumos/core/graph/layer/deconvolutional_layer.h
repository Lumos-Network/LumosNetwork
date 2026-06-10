#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "image.h"
#include "active.h"
#include "bias.h"
#include "gemm.h"
#include "cpu.h"

#include "deconvolutional_layer_gpu.h"

#ifdef __cplusplus
extern "C"{
#endif

Layer *make_deconvolutional_layer(int filters, int ksize, int stride, int pad, int dilation, int bias, char *active);

void init_deconvolutional_layer(Layer *l, int w, int h, int c, int subdivision);
void weightinit_deconvolutional_layer(Layer l, FILE *fp);

void forward_deconvolutional_layer(Layer l, int num);
void backward_deconvolutional_layer(Layer l, int num, float *n_delta);
void refresh_deconvolutional_layer_weights(Layer l);
void save_deconvolutional_layer_weights(Layer l, FILE *fp);
void zerograd_deconvolutional_layer(Layer l, int subdivision);

void deconvolutional_constant_kernel_init(Layer l, float x);
void deconvolutional_normal_kernel_init(Layer l, float mean, float std);
void deconvolutional_uniform_kernel_init(Layer l, float min, float max);
void deconvolutional_xavier_normal_kernel_init(Layer l, float gain);
void deconvolutional_xavier_uniform_kernel_init(Layer l, float gain);
void deconvolutional_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity);
void deconvolutional_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity);

void deconvolutional_constant_bias_init(Layer l, float x);
void deconvolutional_normal_bias_init(Layer l, float mean, float std);
void deconvolutional_uniform_bias_init(Layer l, float min, float max);
void deconvolutional_xavier_normal_bias_init(Layer l, float gain);
void deconvolutional_xavier_uniform_bias_init(Layer l, float gain);
void deconvolutional_kaiming_normal_bias_init(Layer l, char *mode);
void deconvolutional_kaiming_uniform_bias_init(Layer l, char *mode);

void deconvolutional_bilinearinterp_init(Layer l);

void deconvolutional_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize);

#ifdef __cplusplus
}
#endif

#endif