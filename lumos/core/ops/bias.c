#include "bias.h"

void add_bias(float *origin, float *bias, int batch, int n, int size)
{
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < size; ++j){
                origin[(b*n + i)*size + j] += bias[i];
            }
        }
    }
}

void scale_bias(float *origin, float *bias, int batch, int n, int size)
{
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < size; ++j){
                origin[(b*n + i)*size + j] *= bias[i];
            }
        }
    }
}

void backward_bias(float *bias_delta, float *delta, int batch, int n, int size)
{
    for (int b = 0; b < batch; ++b){
        for (int i = 0; i < n; ++i){
            float res = 0;
            sum_cpu(delta+(b*n + i)*size, size, &res);
            bias_delta[i] += res;
        }
    }
}
