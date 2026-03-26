#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include <stdlib.h>

#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct node Node;

typedef struct graph{
    int status;
    float *input;
    float *output;
    float *delta;
    float *detect;
    Node *head;
    Node *tail;
} graph, Graph;

struct node{
    Layer *l;
    Node *head;
    Node *next;
};

Graph *create_graph();

void append_layer2grpah(Graph *graph, Layer *l);
void init_graph(Graph *g, int w, int h, int c, int coretype, int subdivision, int group, int optimizer, char *weights_path);
void set_graph(Graph *g, float *space, float *truth, float *loss);
void forward_graph(Graph *g, float *input, int coretype, int subdivision);
void backward_graph(Graph *g, int coretype, int subdivision);
void update_graph(Graph *g, int coretype, float rate, int subdivision);
void refresh_graph(Graph *g, int coretype);
void save_weights(Graph *g, int coretype, FILE *fp);
void zerograd_graph(Graph *g, int subdivision, int coretype);

void SGDOptimizer_graph(Graph *g, int coretype, float rate, int subdivision, float momentum, float dampening, float decay, int nesterov, int maximize);
void AdamOptimizer_graph(Graph *g, int coretype, float rate, int subdivision, float beta1, float beta2, float decay, int amsgrad, int maximize);

#ifdef __cplusplus
}
#endif

#endif