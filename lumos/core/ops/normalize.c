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

void normalize_cpu(float *data, float *mean, float *variance, int num, int features, float *space)
{
    for (int i = 0; i < features; ++i){
        float *data_c = data + i*num;
        float *space_c = space + i*num;
        for (int j = 0; j < num; ++j){
            space_c[j] = (data_c[j] - mean[i]) / (sqrt(variance[i] + .00001f));
        }
    }
}

void gradient_normalize_mean(float *n_delta, float *variance, int num, int features, float *mean_delta)
{
    for (int i = 0; i < features; ++i){
        mean_delta[i] = 0;
        for (int j = 0; j < num; ++j){
            mean_delta[i] += n_delta[i*num+j];
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}

void gradient_normalize_variance(float *n_delta, float *input, float *mean, float *variance, int num, int features, float *variance_delta)
{
    for (int i = 0; i < features; ++i){
        variance_delta[i] = 0;
        for (int j = 0; j < num; ++j){
            variance_delta[i] += n_delta[i*num+j]*(input[i*num+j]-mean[i]);
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

void gradient_normalize_cpu(float *input, float *mean, float *variance, float *mean_delta, float *variance_delta, int num, int features, float *n_delta, float *l_delta)
{
    for (int i = 0; i < features; ++i){
        for (int j = 0; j < num; ++j){
            l_delta[i*num+j] = n_delta[i*num+j] * 1./(sqrt(variance[i] + .00001f)) + variance_delta[i] * 2. * (input[i*num+j] - mean[i]) / (num) + mean_delta[i]/(num);
        }
    }
}

void gradient_scale(float *norm_x, float *mean, float *variance, float *delta, int num, int features, float *space)
{
    for (int i = 0; i < features; ++i){
        float sum = 0;
        for (int j = 0; j < num; ++j){
            sum += norm_x[j] * delta[i*num+j];
        }
        space[i] = sum;
    }
}

void gradient_bias(float *delta, int num, int features, float *space)
{
    for (int i = 0; i < features; ++i){
        float sum = 0;
        for (int j = 0; j < num; ++j){
            sum += delta[i*num+j];
        }
        space[i] = sum;
    }
}
