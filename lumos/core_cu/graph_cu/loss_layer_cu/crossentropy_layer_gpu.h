#ifndef CROSSENTROPY_LAYER_GPU_H
#define CROSSENTROPY_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"
#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"
#include "gemm_gpu.h"
#include "softmax_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_crossentropy_layer_gpu(Layer *l, int w, int h, int c, int subdivision);
void forward_crossentropy_layer_gpu(Layer l, int num);
void backward_crossentropy_layer_gpu(Layer l, int num, float *n_delta);

void zerograd_crossentropy_layer_gpu(Layer l, int subdivision);

#ifdef __cplusplus
}
#endif
#endif
