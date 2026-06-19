#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "layer.h"
#include "cpu.h"
#include "gemm.h"

#include "yolo_layer_gpu.h"
#include "softmax.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_yolo_layer();

void init_yolo_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_yolo_layer(Layer l, int num);
void backward_yolo_layer(Layer l, int num, float *n_delta);

void zerograd_yolo_layer(Layer l, int subdivision);

#ifdef __cplusplus
}
#endif

#endif