#ifndef BIAS_H
#define BIAS_H

#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void add_bias(float *origin, float *bias, int batch, int n, int size);
void scale_bias(float *origin, float *bias, int batch, int n, int size);
void backward_bias(float *bias_delta, float *delta, int batch, int n, int size);

#ifdef __cplusplus
}
#endif

#endif