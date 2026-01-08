#ifndef GLOBAL_MAXPOOL_LAYER_H
#define GLOBAL_MAXPOOL_LAYER_H

#include "layer.h"
#include "cpu.h"
#include "pooling.h"

#include "global_maxpool_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_global_maxpool_layer();

void init_global_maxpool_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_global_maxpool_layer(Layer l, int num);
void backward_global_maxpool_layer(Layer l, int num, float *n_delta);

void free_global_maxpool_layer(Layer l);

#ifdef __cplusplus
}
#endif

#endif
