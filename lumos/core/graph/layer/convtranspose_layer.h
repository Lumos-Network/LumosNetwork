#ifndef CONVTRANSPOSE_LAYER_H
#define CONVTRANSPOSE_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"

#include "convtranspose_layer_gpu.h"

#ifdef __cplusplus
extern "C"{
#endif

void convpadding(float *img, int height, int width, int channel, int ksize, int stride, int pad, float *space);

Layer *make_convtranspose_layer(int filters, int ksize, int stride, int pad, int bias, char *active);

void init_convtranspose_layer(Layer *l, int w, int h, int c, int subdivision);
void weightinit_convtranspose_layer(Layer l, FILE *fp);

void forward_convtranspose_layer(Layer l, int num);
void backward_convtranspose_layer(Layer l, int num, float *n_delta);
void refresh_convtranspose_layer_weights(Layer l);
void save_convtranspose_layer_weights(Layer l, FILE *fp);
void zerograd_convtranspose_layer(Layer l, int subdivision);

void convtranspose_constant_kernel_init(Layer l, float x);
void convtranspose_normal_kernel_init(Layer l, float mean, float std);
void convtranspose_uniform_kernel_init(Layer l, float min, float max);
void convtranspose_xavier_normal_kernel_init(Layer l, float gain);
void convtranspose_xavier_uniform_kernel_init(Layer l, float gain);
void convtranspose_kaiming_normal_kernel_init(Layer l, float a, char *mode, char *nonlinearity);
void convtranspose_kaiming_uniform_kernel_init(Layer l, float a, char *mode, char *nonlinearity);

void convtranspose_constant_bias_init(Layer l, float x);
void convtranspose_normal_bias_init(Layer l, float mean, float std);
void convtranspose_uniform_bias_init(Layer l, float min, float max);
void convtranspose_xavier_normal_bias_init(Layer l, float gain);
void convtranspose_xavier_uniform_bias_init(Layer l, float gain);
void convtranspose_kaiming_normal_bias_init(Layer l, char *mode);
void convtranspose_kaiming_uniform_bias_init(Layer l, char *mode);

void convtranspose_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize);

#ifdef __cplusplus
}
#endif

#endif