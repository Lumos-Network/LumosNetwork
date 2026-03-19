#ifndef SHORTCUT_LAYER_GPU_H
#define SHORTCUT_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdlib.h>
#include <stdio.h>

#include "gpu.h"
#include "cpu_gpu.h"
#include "layer.h"
#include "shortcut_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_shortcut_layer_gpu(Layer *l, int w, int h, int c, int subdivision);

void forward_shortcut_layer_gpu(Layer l, int num);
void backward_shortcut_layer_gpu(Layer l, int num, float *n_delta);

void zerograd_shortcut_layer_gpu(Layer l, int subdivision);

#ifdef __cplusplus
}
#endif
#endif
