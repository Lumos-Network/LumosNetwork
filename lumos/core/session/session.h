#ifndef SESSION_H
#define SESSION_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "graph.h"
#include "text_f.h"
#include "binary_f.h"
#include "image.h"
#include "progress_bar.h"
#include "optimize.h"
#include "binary_f.h"

#ifdef __cplusplus
extern "C" {
#endif 

typedef struct session{
    Graph *graph;

    int optimizer;
    int coretype;
    int epoch;
    int batch;
    int subdivision;

    int width;
    int height;
    int channel;

    float *loss;

    float learning_rate;
    size_t workspace_size;

    float *workspace;
    float *input;

    float *truth;
    int truth_num;

    int train_data_num;
    char **train_data_paths;
    char **train_label_paths;

    char *weights_path;

    LrScheduler *lrscheduler;

    float momentum;
    float dampening;
    float decay;
    int nesterov;
    int maximize;

    int resize;
    int row;
    int col;

    int normalize;
    float *mean;
    float *std;
} Session;

Session *create_session(Graph *graph, int h, int w, int c, int truth_num, char *type, char *path);
void init_session(Session *sess, char *data_path, char *label_path);

void bind_train_data(Session *sess, char *path);
void bind_train_label(Session *sess, char *path);

void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);
void set_detect_params(Session *sess);
void create_workspace(Session *sess);
void train(Session *sess);
void detect_classification(Session *sess);

void load_train_data(Session *sess, int index);
void load_train_data_binary(Session *sess, int index);
void load_train_label(Session *sess, int index);

void lr_scheduler_step(Session *sess, int step_size, float gamma);
void lr_scheduler_multistep(Session *sess, int *milestones, int num, float gamma);
void lr_scheduler_exponential(Session *sess, float gamma);
void lr_scheduler_cosineannealing(Session *sess, int T_max, float lr_min);

void init_constant_kernel(Layer *l, float x);
void init_normal_kernel(Layer *l, float mean, float std);
void init_uniform_kernel(Layer *l, float min, float max);
void init_xavier_normal_kernel(Layer *l, float gain);
void init_xavier_uniform_kernel(Layer *l, float gain);
void init_kaiming_normal_kernel(Layer *l, float a, char *mode, char *nonlinearity);
void init_kaiming_uniform_kernel(Layer *l, float a, char *mode, char *nonlinearity);

void init_constant_bias(Layer *l, float x);
void init_normal_bias(Layer *l, float mean, float std);
void init_uniform_bias(Layer *l, float min, float max);
void init_xavier_normal_bias(Layer *l, float gain);
void init_xavier_uniform_bias(Layer *l, float gain);
void init_kaiming_normal_bias(Layer *l, char *mode);
void init_kaiming_uniform_bias(Layer *l, char *mode);

void SGDOptimizer_sess(Session *sess, float momentum, float dampening, float decay, int nesterov, int maximize);

void transform_resize_sess(Session *sess, int height, int width);
void transform_normalize_sess(Session *sess, float *mean, float *std);
void transforms_sess(Session *sess);

#ifdef __cplusplus
}
#endif
#endif
