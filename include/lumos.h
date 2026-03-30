#ifndef LUMOS_H
#define LUMOS_H

#include <stdio.h>
#include <stdlib.h>

#define CPU 0
#define GPU 1

typedef struct session Session;
typedef struct graph Graph;
typedef struct layer Layer;

Session *create_session(Graph *graph, int h, int w, int c, int truth_num, char *type, char *path);
void init_session(Session *sess, char *data_path, char *label_path);
void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate);
void set_detect_params(Session *sess);
void train(Session *sess);
void detect_classification(Session *sess);
void SGDOptimizer_sess(Session *sess, float momentum, float dampening, float decay, int nesterov, int maximize);
void transform_resize_sess(Session *sess, int height, int width);
void transform_normalize_sess(Session *sess, float *mean, float *std);

Graph *create_graph();
void append_layer2grpah(Graph *graph, Layer *l);

Layer *make_avgpool_layer(int ksize, int stride, int pad);
Layer *make_connect_layer(int output, int bias, char *active);
Layer *make_convolutional_layer(int filters, int ksize, int stride, int pad, int bias, char *active);
Layer *make_dropout_layer(float probability);
Layer *make_normalization_layer(float momentum, int affine, char *active);
Layer *make_shortcut_layer(Layer *shortcut, int shortcuttype, char *active);
Layer *make_maxpool_layer(int ksize, int stride, int pad);
Layer *make_softmax_layer(int group);
Layer *make_logsoftmax_layer(int group);

Layer *make_mse_layer(int group);
Layer *make_mae_layer(int group);
Layer *make_ce_layer(int group);
Layer *make_nll_layer(int group);
Layer *make_crossentropy_layer(int group);

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

#endif