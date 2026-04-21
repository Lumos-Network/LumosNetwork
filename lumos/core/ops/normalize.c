#include "normalize.h"

void normalize_mean(float *data, int num, int features, int subdivision, float *mean)
{
    for (int i = 0; i < features; ++i){
        mean[i] = 0;
        for (int j = 0; j < subdivision; ++j){
            for (int k = 0; k < num; ++k){
                mean[i] += data[j*num*features+i*num+k];
            }
        }
        mean[i] /= subdivision*num;
    }
}

void normalize_variance(float *data, int num, int features, int subdivision, float *mean, float *variance)
{
    for (int i = 0; i < features; ++i){
        variance[i] = 0;
        for (int j = 0; j < subdivision; ++j){
            for (int k = 0; k < num; ++k){
                variance[i] += pow(data[j*num*features+i*num+k]-mean[i], 2);
            }
        }
        variance[i] /= subdivision*num;
    }
}

void normalize_cpu(float *data, float *mean, float *variance, int num, int features, int subdivision, float *space)
{
    for (int i = 0; i < subdivision; ++i){
        for (int j = 0; j < features; ++j){
            for (int k = 0; k < num; ++k){
                int index = i*num*features+j*num+k;
                space[index] = (data[index] - mean[j]) / sqrt(variance[j] + .0000001f);
            }
        }
    }
}

void gradient_normalize_mean(float *n_delta, float *variance, int num, int features, int subdivision, float *mean_delta)
{
    for (int i = 0; i < features; ++i){
        mean_delta[i] = 0;
        for (int j = 0; j < subdivision; ++j){
            for (int k = 0; k < num; ++k){
                int index = j*features*num + i*num + k;
                mean_delta[i] += n_delta[index];
            }
            mean_delta[i] *= (-1./sqrt(variance[i] + .0000001f));
        }
    }
}

void gradient_normalize_variance(float *n_delta, float *input, float *mean, float *variance, int num, int features, int subdivision, float *variance_delta)
{
    for (int i = 0; i < features; ++i){
        variance_delta[i] = 0;
        for (int j = 0; j < subdivision; ++j){
            for (int k = 0; k < num; ++k){
                int index = j*features*num + i*num + k;
                variance_delta[i] += n_delta[index]*(input[index]-mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .0000001f, (float)(-3./2.));
    }
}

void gradient_normalize_cpu(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int num, int features, int subdivision, float *n_delta, float *l_delta)
{
    for (int i = 0; i < subdivision; ++i){
        for (int j = 0; j < features; ++j){
            for (int k = 0; k < num; ++k){
                int index = i*features*num + j*num + k;
                l_delta[index] = n_delta[index] * 1./(sqrt(variance[j] + .0000001f)) + variance_delta[j] * 2. * (input[index] - mean[j]) / (num*subdivision) + mean_delta[j]/(num*subdivision);
            }
        }
    }
}

void gradient_scale(float *norm_x, float *n_delta, int num, int features, int subdivision, float *space)
{
    for (int i = 0; i < features; ++i){
        float sum = 0;
        for (int j = 0; j < subdivision; ++j){
            for (int k = 0; k < num; ++k){
                int index = k + num*(i + features*j);
                sum += n_delta[index] * norm_x[index];
            }
        }
        space[i] += sum;
    }
}
