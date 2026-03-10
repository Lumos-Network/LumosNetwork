#ifndef CROSSENTROPY_LAYER_H
#define CROSSENTROPY_LAYER_H

#include "layer.h"
#include "cpu.h"
#include "gemm.h"

#include "crossentropy_layer_gpu.h"
#include "softmax.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_crossentropy_layer(int group);

void init_crossentropy_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_crossentropy_layer(Layer l, int num);
void backward_crossentropy_layer(Layer l, int num, float *n_delta);

void free_crossentropy_layer(Layer l);

#ifdef __cplusplus
}
#endif

#endif