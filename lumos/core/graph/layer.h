#ifndef LAYER_H
#define LAYER_H

#include <stdlib.h>
#include <stdio.h>

#include "active.h"
#include "active_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CPU 0
#define GPU 1

#define SGD 0
#define ADAM 1

#define BASIC 0
#define SHORT 1

typedef enum {
    CONVOLUTIONAL, CONNECT, IM2COL, MAXPOOL, AVGPOOL, GLOBALMAX, GLOBALAVG, \
    DROPOUT, SOFTMAX, LOGSOFTMAX, SHORTCUT, NORMALIZE, \
    MSE, MAE, CE, NLL, CROSSENTROPY
} LayerType;

typedef enum {
    CONSTANT_I, NORMAL_I, UNIFORM_I, XAVIER_NORMAL_I, XAVIER_UNIFORM_I, KAIMING_NORMAL_I, KAIMING_UNIFORM_I
} InitType;

typedef struct initcptkernel{
    InitType initype;
    float x;
    float mean;
    float std;
    float a;
    float min;
    float max;
    char *mode;
    char *nonlinearity;
} initcptkernel, InitCptKernel;

typedef struct initcptbias{
    InitType initype;
    float x;
    float mean;
    float std;
    float a;
    float min;
    float max;
    char *mode;
    char *nonlinearity;
} initcptbias, InitCptBias;

typedef struct initcptkernel InitCptKernel;
typedef struct layer Layer;

typedef void (*forward)  (struct layer, int);
typedef void (*backward) (struct layer, int, float*);
typedef void (*update) (struct layer, float, int, float*);
typedef void (*refresh) (struct layer);
typedef forward Forward;
typedef backward Backward;
typedef update Update;
typedef refresh Refresh;

typedef void (*forward_gpu)  (struct layer, int);
typedef void (*backward_gpu) (struct layer, int, float*);
typedef void (*update_gpu) (struct layer, float, int, float*);
typedef void (*refresh_gpu) (struct layer);
typedef forward_gpu ForwardGpu;
typedef backward_gpu BackwardGpu;
typedef update_gpu UpdateGpu;
typedef refresh_gpu RefreshGpu;

typedef void (*sgdoptimizer) (struct layer, float, float, float, float, int, int);
typedef void (*sgdoptimizer_gpu) (struct layer, float, float, float, float, int, int);
typedef sgdoptimizer SGDOptimizer;
typedef sgdoptimizer_gpu SGDOptimizerGpu;
typedef void (*adamoptimizer) (struct layer, float, float, float, float, int, int, float*);
typedef void (*adamoptimizer_gpu) (struct layer, float, float, float, float, int, int, float*);
typedef adamoptimizer AdamOptimizer;
typedef adamoptimizer_gpu AdamOptimizerGpu;

typedef void (*initialize) (struct layer *, int, int, int, int);
typedef void (*initialize_gpu) (struct layer *, int, int, int, int);
typedef initialize Initialize;
typedef initialize_gpu InitializeGpu;

typedef void (*weightinit) (struct layer, FILE*);
typedef weightinit WeightInit;
typedef void (*weightinit_gpu) (struct layer, FILE*);
typedef weightinit_gpu WeightInitGpu;

typedef void (*saveweights) (struct layer, FILE*);
typedef saveweights SaveWeights;
typedef void (*saveweights_gpu) (struct layer, FILE*);
typedef saveweights_gpu SaveWeightsGpu;

typedef void (*zerograd_layer) (struct layer, int);
typedef zerograd_layer ZeroGradLayer;
typedef void (*zerograd_layer_gpu) (struct layer, int);
typedef zerograd_layer_gpu ZeroGradLayerGpu;

struct layer{
    LayerType type;
    int optimizer;
    int status;
    int input_h;
    int input_w;
    int input_c;
    int output_h;
    int output_w;
    int output_c;

    int inputs;
    int outputs;

    int workspace_size;
    int truth_num;

    float *input;
    float *output;
    float *delta;
    float *truth;
    float *loss;
    float *workspace;
    float *kernel_weights_delta;
    float *bias_delta;

    int *maxpool_index;
    //为社么是指针
    int *dropout_rand;

    int filters;
    int ksize;
    int stride;
    int pad;
    int group;

    int bias;
    // dropout 占比
    float probability;
    float *detect; //预测值
    Layer *shortcut;
    int shortcut_type;

    float *kernel_weights;
    float *bias_weights;

    float *update_kernel_weights;
    float *update_bias_weights;

    int affine;
    float *mean;
    float *variance;
    float *rolling_mean;
    float *rolling_variance;
    float *mean_delta;
    float *variance_delta;
    float *norm_x;
    float bn_momentum;

    float *momentum_kernel_v;
    float *momentum_bias_v;

    int step_t;
    float *exp_avg_kernel;
    float *exp_avg_sq_kernel;
    float *exp_avg_bias;
    float *exp_avg_sq_bias;
    float *exp_avg_sq_kernel_max;
    float *exp_avg_sq_bias_max;

    Forward forward;
    Backward backward;
    Update update;
    Refresh refresh;

    ForwardGpu forwardgpu;
    BackwardGpu backwardgpu;
    UpdateGpu updategpu;
    RefreshGpu refreshgpu;

    SGDOptimizer sgdoptimizer;
    SGDOptimizerGpu sgdoptimizergpu;

    AdamOptimizer adamoptimizer;
    AdamOptimizerGpu adamoptimizergpu;

    Initialize initialize;
    InitializeGpu initializegpu;

    WeightInit weightinit;
    WeightInitGpu weightinitgpu;

    Activation active;
    SaveWeights saveweights;
    SaveWeightsGpu saveweightsgpu;

    ZeroGradLayer zerogradlayer;
    ZeroGradLayerGpu zerogradlayergpu;

    InitCptKernel *initcptkernel;
    InitCptBias *initcptbias;
};

#ifdef __cplusplus
}
#endif

#endif