#ifndef INTERPOLATE_LAYER_H
#define INTERPOLATE_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layer.h"
#include "cpu.h"

#include "interpolate_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void interpolate(float *img, int height, int width, int channel, int row, int col, float *space);
void interpolate_gradient(float *img, int row, int col, int channel, int height, int width, float *space);

Layer *make_interpolate_layer(int height, int width);

void init_interpolate_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_interpolate_layer(Layer l, int num);
void backward_interpolate_layer(Layer l, int num, float *n_delta);

void zerograd_interpolate_layer(Layer l, int subdivision);

#ifdef __cplusplus
}
#endif

#endif