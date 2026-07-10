#ifndef RANDOM_GPU_H
#define RANDOM_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_random(float *space, int num);

#ifdef __cplusplus
}
#endif
#endif
