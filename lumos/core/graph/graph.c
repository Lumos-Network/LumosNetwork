#include "graph.h"

Graph *create_graph()
{
    Graph *graph = malloc(sizeof(Graph));
    graph->head = NULL;
    graph->tail = NULL;
    fprintf(stderr, "[Lumos]         Module Structure\n");
    return graph;
}

void append_layer2grpah(Graph *graph, Layer *l)
{
    Node *layer = malloc(sizeof(Node));
    if (graph->tail){
        Node *tail = graph->tail;
        tail->next = layer;
    }
    layer->l = l;
    layer->next = NULL;
    layer->head = graph->tail;
    graph->tail = layer;
    if (graph->head == NULL) graph->head = layer;
}

void init_graph(Graph *g, int w, int h, int c, int coretype, int subdivision, int optimizer, char *weights_path, float *input)
{
    fprintf(stderr, "\nStart To Init Graph\n");
    fprintf(stderr, "[Lumos]                     Inputs         Outputs\n");
    Node *layer = g->head;
    Layer *l;
    FILE *fp = NULL;
    if (weights_path){
        fp = fopen(weights_path, "rb");
    }
    for (;;){
        if (layer){
            l = layer->l;
            l->optimizer = optimizer;
            l->input = input;
            if (coretype == GPU){
                l->initializegpu(l, w, h, c, subdivision);
                if (l->weightinitgpu) l->weightinitgpu(*l, fp);
            } else {
                l->initialize(l, w, h, c, subdivision);
                if (l->weightinit) l->weightinit(*l, fp);
            }
        } else {
            if (coretype == GPU) cudaMalloc((void**)&g->detect, subdivision*l->inputs*sizeof(float));
            else g->detect = calloc(subdivision*l->inputs, sizeof(float));
            break;
        }
        layer = layer->next;
        w = l->output_w;
        h = l->output_h;
        c = l->output_c;
        input = l->output;
    }
    if (fp){
        fclose(fp);
    }
}

void set_graph(Graph *g, float *space, int *truth, float *loss)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            l->truth = truth;
            l->loss = loss;
            l->workspace = space;
            l->detect = g->detect;
        } else {
            break;
        }
        layer = layer->next;
    }
}

void forward_graph(Graph *g, int coretype, int subdivision)
{
    Node *layer = g->head;
    Layer *l;
    int i = 0;
    for (;;){
        if (layer){
            l = layer->l;
            l->status = g->status;
            if (coretype == GPU){
                l->forwardgpu(*l, subdivision);
            } else {
                l->forward(*l, subdivision);
            }
            if (i == 0){
                FILE *fp = fopen("./backup/in_c", "wb");
                fwrite(l->input, sizeof(float), l->inputs*subdivision, fp);
                fclose(fp);
            }
        } else {
            break;
        }
        i += 1;
        layer = layer->next;
    }
}

void backward_graph(Graph *g, int coretype, int subdivision)
{
    Node *layer = g->tail;
    Layer *l;
    float *n_delta;
    int i = 0;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU){
                l->backwardgpu(*l, subdivision, n_delta);
            } else {
                l->backward(*l, subdivision, n_delta);
            }
            // if (i == 2){
            //     FILE *fp = fopen("./backup/grad_c", "wb");
            //     fwrite(l->delta, sizeof(float), subdivision*l->inputs, fp);
            //     fclose(fp);
            // }
        } else {
            break;
        }
        i += 1;
        layer = layer->head;
        n_delta = l->delta;
    }
}

void refresh_graph(Graph *g, int coretype)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU && l->refreshgpu) l->refreshgpu(*l);
            if (coretype == CPU && l->refresh) l->refresh(*l);
        } else {
            break;
        }
        layer = layer->next;
    }
}

void save_weights(Graph *g, int coretype, FILE *fp)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU && l->saveweightsgpu) l->saveweightsgpu(*l, fp);
            if (coretype == CPU && l->saveweights) l->saveweights(*l, fp);
        } else {
            break;
        }
        layer = layer->next;
    }
}

void zerograd_graph(Graph *g, int subdivision, int coretype)
{
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU && l->zerogradlayergpu) l->zerogradlayergpu(*l, subdivision);
            if (coretype == CPU && l->zerogradlayer) l->zerogradlayer(*l, subdivision);
        } else {
            break;
        }
        layer = layer->next;
    }
}

void SGDOptimizer_graph(Graph *g, int coretype, float rate, float momentum, float dampening, float decay, int nesterov, int maximize)
{
    Node *layer = g->tail;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU && l->sgdoptimizergpu) l->sgdoptimizergpu(*l, rate, momentum, dampening, decay, nesterov, maximize);
            if (coretype == CPU && l->sgdoptimizer) l->sgdoptimizer(*l, rate, momentum, dampening, decay, nesterov, maximize);
        } else {
            break;
        }
        layer = layer->head;
    }
}

void AdamOptimizer_graph(Graph *g, int coretype, float rate, float beta1, float beta2, float decay, int amsgrad, int maximize)
{
    Node *layer = g->tail;
    Layer *l;
    float *n_delta;
    for (;;){
        if (layer){
            l = layer->l;
            if (coretype == GPU && l->sgdoptimizergpu) l->adamoptimizergpu(*l, rate, beta1, beta2, decay, amsgrad, maximize, n_delta);
            if (coretype == CPU && l->sgdoptimizer) l->adamoptimizer(*l, rate, beta1, beta2, decay, amsgrad, maximize, n_delta);
        } else {
            break;
        }
        layer = layer->head;
        n_delta = l->delta;
    }
}
