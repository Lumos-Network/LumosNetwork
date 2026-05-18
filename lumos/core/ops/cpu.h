#ifndef CPU_H
#define CPU_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C"{
#endif

// offset=1 为正常偏移
void fill_cpu(float *data, int len, float x, int offset);
void multy_cpu(float *data, int len, float x, int offset);
void add_cpu(float *data, int len, float x, int offset);

void min_cpu(float *data, int num, float *space);
void max_cpu(float *data, int num, float *space);
void sum_cpu(float *data, int num, float *space);
void mean_cpu(float *data, int num, float *space);
void variance_cpu(float *data, float mean, int num, float *space);

void matrix_add_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_subtract_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_multiply_cpu(float *data_a, float *data_b, int num, float *space);
void matrix_divide_cpu(float *data_a, float *data_b, int num, float *space);

void saxpy_cpu(float *data_a, float *data_b, int num, float x, float *space);
void sum_channel_cpu(float *data, int h, int w, int c, float ALPHA, float *space);

void lerp_cpu(float *data_a, float *data_b, int num, float x, float *space);
void lerp2_cpu(float *data_a, float *data_b, int num, float x, float *space);
void maximum_cpu(float *data_a, float *data_b, int num, float *space);
void addcdiv_cpu(float *input, float *data_a, float *data_b, float x, int num, float *space);

void one_hot_encoding(int n, int label, float *space);
int find_max(float *data, int num);

void copy_nums(int **shapes, int num, int dims, int cdim, int *cnums);
void acc_multy_int(int *data, int num, int index, int flag, int *res);
void array_cat(float **datas, int **shapes, int num, int dims, int cdim, float *space);
void array_split(float *data, int **shapes, int num, int dims, int cdim, float **space);

#ifdef __cplusplus
}
#endif

#endif