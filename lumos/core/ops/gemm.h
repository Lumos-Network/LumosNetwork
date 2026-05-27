#ifndef GEMM_H
#define GEMM_H

#include "cpu.h"

#ifdef __cplusplus
extern "C"{
#endif
// flag=1 C制零后运算
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc);

void gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc);
void gemm_tn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc);
void gemm_nt(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc);
void gemm_tt(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc);

#ifdef __cplusplus
}
#endif

#endif