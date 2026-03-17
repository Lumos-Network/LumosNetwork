#ifndef NORMALIZE_GPU_H
#define NORMALIZE_GPU_H

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

void normalize_mean_gpu(float *data, int num, int features, int subdivision, float *mean);
void normalize_variance_gpu(float *data, int num, int features, int subdivision, float *mean, float *variance);
void normalize_gpu(float *data, float *mean, float *variance, int num, int features, float *space);

void gradient_normalize_mean_gpu(float *n_delta, float *variance, int num, int features, float *mean_delta);
void gradient_normalize_variance_gpu(float *n_delta, float *input, float *mean, float *variance, int num, int features, float *variance_delta);
void gradient_normalize_gpu(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int num, int features, float *n_delta, float *l_delta);

void update_scale_gpu(float *output, float *delta, int num, int features, float rate, float *space);
void update_bias_gpu(float *delta, int num, int features, float rate, float *space);

#ifdef __cplusplus
}
#endif
#endif
