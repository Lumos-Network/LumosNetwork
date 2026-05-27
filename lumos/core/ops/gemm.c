#include "gemm.h"

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    #pragma omp parallel for
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            register float temp = ALPHA * A[i * K + j];
            for (int k = 0; k < N; ++k){
                C[i * N + k] += temp * B[j * N + k];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    #pragma omp parallel for
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            register float temp = ALPHA * A[i * K + j];
            for (int k = 0; k < N; ++k){
                C[i * N + k] += temp * B[k * K + j];
            }
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    #pragma omp parallel for
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            register float temp = ALPHA * A[j * M + i];
            for (int k = 0; k < N; ++k){
                C[i * N + k] += temp * B[j * N + k];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    #pragma omp parallel for
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < K; ++j){
            register float temp = ALPHA * A[j * M + i];
            for (int k = 0; k < N; ++k){
                C[i * N + k] += temp * B[k * K + j];
            }
        }
    }
}
