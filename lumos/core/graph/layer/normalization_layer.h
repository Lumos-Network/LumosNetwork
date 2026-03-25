#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "cpu.h"
#include "bias.h"
#include "normalize.h"

#include "normalization_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_normalization_layer(float momentum, int affine, char *active);
void init_normalization_layer(Layer *l, int w, int h, int c, int subdivision);
void weightinit_normalization_layer(Layer l, FILE *fp);

void forward_normalization_layer(Layer l, int num);
void backward_normalization_layer(Layer l, int num, float *n_delta);
void normalization_layer_SGDOptimizer(Layer l, float rate, float momentum, float dampening, float decay, int nesterov, int maximize, int num, float *n_delta);
void refresh_normalization_layer_weights(Layer l);

void save_normalization_layer_weights(Layer l, FILE *fp);
void zerograd_normalization_layer(Layer l, int subdivision);

#ifdef __cplusplus
}
#endif

#endif