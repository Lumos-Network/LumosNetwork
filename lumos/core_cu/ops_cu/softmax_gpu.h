#ifndef SOFTMAX_GPU_H
#define SOFTMAX_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void softmax_gpu(float *data, int num, float *space, float *ALPHA);
void softmax_gradient_gpu(float *data, int num, float *space, float *ALPHA);
void softmax_exp_sum_gpu(float *data, int num, float *workspace, float *space);

void log_softmax_gpu(float *data, int num, float *space, float *ALPHA);
void log_softmax_gradient_gpu(float *data, int num, float *space, float *ALPHA);

#ifdef __cplusplus
}
#endif
#endif
