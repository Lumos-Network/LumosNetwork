#ifndef NLL_LAYER_H
#define NLL_LAYER_H

#include "layer.h"
#include "cpu.h"
#include "gemm.h"

#include "nll_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_nll_layer(int group);

void init_nll_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_nll_layer(Layer l, int num);
void backward_nll_layer(Layer l, int num, float *n_delta);

void zerograd_nll_layer(Layer l, int subdivision);

#ifdef __cplusplus
}
#endif

#endif