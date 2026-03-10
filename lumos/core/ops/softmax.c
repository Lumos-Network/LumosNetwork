#include "softmax.h"

void softmax(float *data, int num, float *space)
{
    float M = 0;
    float res = 0;
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        res += exp(data[i] - M);
    }
    for (int i = 0; i < num; ++i){
        space[i] = exp(data[i] - M) / res;
    }
}

void softmax_gradient(float *data, int num, float *space)
{
    float M = 0;
    float res = 0;
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        res += exp(data[i] - M);
    }
    for (int i = 0; i < num; ++i){
        float x = exp(data[i] - M);
        space[i] = (res + x) * x;
    }
}

void log_softmax(float *data, int num, float *space)
{
    float M = 0;
    float res = 0;
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        res += exp(data[i] - M);
    }
    for (int i = 0; i < num; ++i){
        space[i] = (data[i] - M) - log(res);
    }
}

void log_softmax_gradient(float *data, int num, float *space)
{
    float M = 0;
    float res = 0;
    max_cpu(data, num, &M);
    for (int i = 0; i < num; ++i){
        space[i] = exp(data[i] - M);
        res += space[i];
    }
    for (int i = 0; i < num; ++i){
        space[i] = 1 - exp(space[i]) / res;
    }
}
