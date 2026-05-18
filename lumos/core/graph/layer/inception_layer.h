#ifndef INCEPTION_LAYER_H
#define INCEPTION_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"

#include "inception_layer_gpu.h"

#ifdef __cplusplus
extern "C"{
#endif

Layer *make_inception_layer(Layer **inception, int num, int dim);
void init_inception_layer(Layer *l, int w, int h, int c, int subdivision);

void forward_inception_layer(Layer l, int num);
void backward_inception_layer(Layer l, int num, float *n_delta);

void zerograd_inception_layer(Layer l, int subdivision);

#ifdef __cplusplus
}
#endif

#endif
