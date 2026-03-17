#include "session.h"

Session *create_session(Graph *graph, int h, int w, int c, int truth_num, char *type, char *path)
{
    Session *sess = malloc(sizeof(Session));
    sess->graph = graph;
    if (0 == strcmp(type, "gpu")){
        sess->coretype = GPU;
    } else {
        sess->coretype = CPU;
    }
    sess->height = h;
    sess->width = w;
    sess->channel = c;
    sess->truth_num = truth_num;
    sess->weights_path = path;
    sess->resize = 0;
    sess->normalize = 0;
    return sess;
}

void init_session(Session *sess, char *data_path, char *label_path)
{
    bind_train_data(sess, data_path);
    bind_train_label(sess, label_path);
    init_graph(sess->graph, sess->width, sess->height, sess->channel, sess->coretype, sess->subdivision, sess->truth_num, sess->optimizer, sess->weights_path);
    create_workspace(sess);
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->input, sess->subdivision*sess->width*sess->height*sess->channel*sizeof(float));
        cudaMalloc((void**)&sess->truth, sess->subdivision*sess->truth_num*sizeof(float));
        cudaMalloc((void**)&sess->loss, sizeof(float));
    } else {
        sess->input = calloc(sess->subdivision*sess->width*sess->height*sess->channel, sizeof(float));
        sess->truth = calloc(sess->subdivision*sess->truth_num, sizeof(float));
        sess->loss = calloc(1, sizeof(float));
    }
    set_graph(sess->graph, sess->workspace, sess->truth, sess->loss);
    transforms_sess(sess);
    bind_train_data(sess, "./backup/train.txt");
}

void bind_train_data(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    char *data_path = NULL;
    sess->train_data_num = lines;
    sess->train_data_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        data_path = tmp+index[i+1];
        strip(data_path, '\n');
        sess->train_data_paths[i] = data_path;
    }
    free(index);
    fprintf(stderr, "\nGet Train Data List From %s\n", path);
}

void bind_train_label(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    char *label_path = NULL;
    sess->train_label_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        label_path = tmp+index[i+1];
        strip(label_path, '\n');
        sess->train_label_paths[i] = label_path;
    }
    free(index);
    fprintf(stderr, "\nGet Label List From %s\n", path);
}

void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate)
{
    sess->epoch = epoch;
    sess->batch = batch;
    sess->subdivision = subdivision;
    sess->learning_rate = learning_rate;
}

void set_detect_params(Session *sess)
{
    sess->epoch = 1;
    sess->batch = 1;
    sess->subdivision = 1;
}

void create_workspace(Session *sess)
{
    Graph *graph = sess->graph;
    Node *layer = graph->head;
    Layer *l;
    int max = -1;
    for (;;){
        if (layer){
            l = layer->l;
            if (l->workspace_size > max) max = l->workspace_size;
        } else {
            break;
        }
        layer = layer->next;
    }
    if (max <= 0) return;
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->workspace, max*sizeof(float));
    } else {
        sess->workspace = calloc(max, sizeof(float));
    }
}

void load_train_data(Session *sess, int index)
{
    int h[1], w[1], c[1];
    float *im;
    int offset_i = 0;
    float *input = (float*)calloc(sess->subdivision*sess->width*sess->height*sess->channel, sizeof(float));
    for (int i = index; i < index + sess->subdivision; ++i){
        char *data_path = sess->train_data_paths[i];
        im = load_image_data(data_path, w, h, c);
        resize_im(im, h[0], w[0], c[0], sess->height, sess->width, input + offset_i);
        offset_i += sess->height * sess->width * sess->channel;
        free(im);
    }
    if (sess->coretype == GPU){
        cudaMemcpy(sess->input, input, sess->subdivision*sess->width*sess->height*sess->channel*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        memcpy(sess->input, input, sess->subdivision*sess->width*sess->height*sess->channel*sizeof(float));
    }
    free(input);
}

void load_train_data_binary(Session *sess, int index)
{
    int offset_i = 0;
    float *input = (float*)calloc(sess->subdivision*sess->width*sess->height*sess->channel, sizeof(float));
    for (int i = index; i < index + sess->subdivision; ++i){
        char *data_path = sess->train_data_paths[i];
        FILE *fp = fopen(data_path, "rb");
        fread(input+offset_i, sizeof(float), sess->height*sess->width*sess->channel, fp);
        fclose(fp);
        offset_i += sess->height * sess->width * sess->channel;
    }
    if (sess->coretype == GPU){
        cudaMemcpy(sess->input, input, sess->subdivision*sess->width*sess->height*sess->channel*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        memcpy(sess->input, input, sess->subdivision*sess->width*sess->height*sess->channel*sizeof(float));
    }
    free(input);
}

void load_train_label(Session *sess, int index)
{
    float *truth = calloc(sess->subdivision*sess->truth_num, sizeof(float));
    for (int i = index; i < index + sess->subdivision; ++i){
        float *truth_i = truth + (i - index) * sess->truth_num;
        char *label_path = sess->train_label_paths[i];
        strip(label_path, ' ');
        void **labels = load_label_txt(label_path);
        int *lindex = (int*)labels[0];
        char *tmp = (char*)labels[1];
        for (int j = 0; j < sess->truth_num; ++j){
            truth_i[j] = (float)atoi(tmp+lindex[j+1]);
        }
        free(lindex);
        free(tmp);
        free(labels);
    }
    if (sess->coretype == GPU){
        cudaMemcpy(sess->truth, truth, sess->truth_num*sess->subdivision*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        memcpy(sess->truth, truth, sess->truth_num*sess->subdivision*sizeof(float));
    }
    free(truth);
}

void train(Session *sess)
{
    fprintf(stderr, "\nSession Start To Running\n");
    float rate = -sess->learning_rate;
    Graph *g = sess->graph;
    g->status = 1;
    for (int i = 0; i < sess->epoch; ++i){
        fprintf(stderr, "\n\nEpoch %d/%d\n", i + 1, sess->epoch);
        int sub_epochs = (int)(sess->train_data_num / sess->batch);
        int sub_batchs = (int)(sess->batch / sess->subdivision);
        float loss[2] = {0, 0};
        for (int j = 0; j < sub_epochs; ++j){
            for (int k = 0; k < sub_batchs; ++k){
                if (j * sess->batch + k * sess->subdivision + sess->subdivision > sess->train_data_num) break;
                load_train_data_binary(sess, j * sess->batch + k * sess->subdivision);
                load_train_label(sess, j * sess->batch + k * sess->subdivision);
                forward_graph(sess->graph, sess->input, sess->coretype, sess->subdivision);
                backward_graph(sess->graph, sess->coretype, sess->subdivision);
                if (sess->optimizer == SGD){
                    SGDOptimizer_graph(sess->graph, sess->coretype, rate, sess->subdivision, sess->momentum, sess->dampening, sess->decay, sess->nesterov, sess->maximize);
                } else {
                    update_graph(sess->graph, sess->coretype, rate, sess->subdivision);
                }
                if (sess->coretype == CPU) {
                    loss[0] += sess->loss[0];
                    loss[1] = sess->loss[0];
                } else{
                    cudaMemcpy(loss+1, sess->loss, sizeof(float), cudaMemcpyDeviceToHost);
                    loss[0] += loss[1];
                }
            }
            refresh_graph(sess->graph, sess->coretype);
            fprintf(stderr, "%d/%d    Loss:%f\n", j, sub_epochs, loss[1]);
        }
        fprintf(stderr, " AvgLoss:%f", loss[0]/(sub_epochs * sub_batchs));
    }
    FILE *fp = fopen("./backup/LW_f", "wb");
    if (fp) {
        save_weights(sess->graph, sess->coretype, fp);
        fclose(fp);
    }
    free_graph(g, sess->coretype);
    fprintf(stderr, "\n\nSession Training Finished\n");
}

void detect_classification(Session *sess)
{
    fprintf(stderr, "\nSession Start To Running\n");
    int num = 0;
    float *truth = NULL;
    float *detect = NULL;
    float *loss = calloc(1, sizeof(float));
    Graph *g = sess->graph;
    g->status = 0;
    Node *layer = g->tail;
    Layer *l = layer->l;
    if (sess->coretype == GPU){
        truth = calloc(sess->truth_num, sizeof(float));
        detect = calloc(sess->truth_num, sizeof(float));
    }
    for (int i = 0; i < sess->train_data_num; ++i){
        load_train_data_binary(sess, i);
        load_train_label(sess, i);
        forward_graph(sess->graph, sess->input, sess->coretype, sess->subdivision);
        if (sess->coretype == GPU){
            cudaMemcpy(truth, l->truth, sess->truth_num*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(detect, l->detect, sess->truth_num*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(loss, sess->loss, sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            truth = l->truth;
            detect = l->detect;
            loss[0] = sess->loss[0];
        }
        fprintf(stderr, "%s\n", sess->train_data_paths[i]);
        fprintf(stderr, "Truth     Detect\n");
        float max = -FLT_MAX;
        int index = -1;
        for (int j = 0; j < sess->truth_num; ++j){
            fprintf(stderr, "%.3f %.3f\n", truth[j], detect[j]);
            if (detect[j] > max){
                index = j;
                max = detect[j];
            }
        }
        if (truth[index] == 1) num += 1;
        fprintf(stderr, "Loss:%.4f\n\n", loss[0]);
    }
    free_graph(g, sess->coretype);
    fprintf(stderr, "Detct Classification: %d/%d  %.2f\n", num, sess->train_data_num, (float)(num)/(float)(sess->train_data_num));
}

void lr_scheduler_step(Session *sess, int step_size, float gamma)
{
    LrScheduler *lrscheduler = make_lrscheduler(SLR, 0, step_size, NULL, 0, 0, gamma);
    sess->lrscheduler = lrscheduler;
}

void lr_scheduler_multistep(Session *sess, int *milestones, int num, float gamma)
{
    LrScheduler *lrscheduler = make_lrscheduler(MLR, num, 0, milestones, 0, 0, gamma);
    sess->lrscheduler = lrscheduler;
}

void lr_scheduler_exponential(Session *sess, float gamma)
{
    LrScheduler *lrscheduler = make_lrscheduler(ELR, 0, 0, NULL, 0, 0, gamma);
    sess->lrscheduler = lrscheduler;
}

void lr_scheduler_cosineannealing(Session *sess, int T_max, float lr_min)
{
    LrScheduler *lrscheduler = make_lrscheduler(CALR, 0, 0, NULL, T_max, lr_min, 0);
    sess->lrscheduler = lrscheduler;
}

void init_constant(Layer *l, float x)
{
    InitCpt *initcpt = malloc(sizeof(InitCpt));
    initcpt->initype = CONSTANT_I;
    initcpt->x = x;
    l->initcpt = initcpt;
}

void init_normal(Layer *l, float mean, float std)
{
    InitCpt *initcpt = malloc(sizeof(InitCpt));
    initcpt->initype = NORMAL_I;
    initcpt->mean = mean;
    initcpt->std = std;
    l->initcpt = initcpt;
}

void init_uniform(Layer *l, float min, float max)
{
    InitCpt *initcpt = malloc(sizeof(InitCpt));
    initcpt->initype = UNIFORM_I;
    initcpt->min = min;
    initcpt->max = max;
    l->initcpt = initcpt;
}

void init_kaiming_normal(Layer *l, float a, char *mode, char *nonlinearity)
{
    InitCpt *initcpt = malloc(sizeof(InitCpt));
    initcpt->initype = KAIMING_NORMAL_I;
    initcpt->a = a;
    initcpt->mode = mode;
    initcpt->nonlinearity = nonlinearity;
    l->initcpt = initcpt;
}

void init_kaiming_uniform(Layer *l, float a, char *mode, char *nonlinearity)
{
    InitCpt *initcpt = malloc(sizeof(InitCpt));
    initcpt->initype = KAIMING_UNIFORM_I;
    initcpt->a = a;
    initcpt->mode = mode;
    initcpt->nonlinearity = nonlinearity;
    l->initcpt = initcpt;
}

void SGDOptimizer_sess(Session *sess, float momentum, float dampening, float decay, int nesterov, int maximize)
{
    sess->momentum = momentum;
    sess->dampening = dampening;
    sess->decay = decay;
    sess->nesterov = nesterov;
    sess->maximize = maximize;
    sess->optimizer = SGD;
}

void transform_resize_sess(Session *sess, int height, int width)
{
    sess->resize = 1;
    sess->row = height;
    sess->col = width;
}

void transform_normalize_sess(Session *sess, float *mean, float *std)
{
    sess->normalize = 1;
    sess->mean = calloc(sess->channel, sizeof(float));
    sess->std = calloc(sess->channel, sizeof(float));
    memcpy(sess->mean, mean, sess->channel*sizeof(float));
    memcpy(sess->std, std, sess->channel*sizeof(float));
}

void transforms_sess(Session *sess)
{
    int h[1], w[1], c[1];
    float *im;
    char path[200];
    FILE *fp = fopen("./backup/train.txt", "w");
    for (int i = 0; i < sess->train_data_num; ++i){
        char *data_path = sess->train_data_paths[i];
        im = load_image_data(data_path, w, h, c);
        if (sess->resize){
            float *new_im = calloc(sess->row*sess->col*sess->channel, sizeof(float));
            resize_im(im, h[0], w[0], c[0], sess->row, sess->col, new_im);
            free(im);
            im = new_im;
            h[0] = sess->row;
            w[0] = sess->col;
            c[0] = sess->channel;
        }
        if (sess->normalize){
            normalize_im(im, h[0], w[0], c[0], sess->mean, sess->std, im);
        }
        sprintf(path, "./backup/data/%d", i);
        FILE *imfp = fopen(path, "wb");
        fwrite(im, sizeof(float), h[0]*w[0]*c[0], imfp);
        fclose(imfp);
        fprintf(fp, "%s\n", path);
        free(im);
    }
    fclose(fp);
}
