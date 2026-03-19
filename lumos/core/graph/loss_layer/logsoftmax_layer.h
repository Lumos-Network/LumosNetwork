#ifndef LOGSOFTMAX_LAYER_H
#define LOGSOFTMAX_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"
#include "softmax.h"
#include "cpu.h"

#include "logsoftmax_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_logsoftmax_layer(int group);

void init_logsoftmax_layer(Layer *l, int w, int h, int c, int subdivision);

void forward_logsoftmax_layer(Layer l, int num);
void backward_logsoftmax_layer(Layer l, int num, float *n_delta);

void zerograd_logsoftmax_layer(Layer l, int subdivision);

#ifdef __cplusplus
}
#endif

#endif