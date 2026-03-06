#include "alexnet_flower.h"

void alexnet_flower(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(96, 11, 4, 2, 1, 0, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 0);
    Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, 0, "relu");
    Layer *l4 = make_maxpool_layer(3, 2, 0);
    Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, 0, "relu");
    Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, 0, "relu");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, 0, "relu");
    Layer *l8 = make_maxpool_layer(3, 2, 0);
    // Layer *l9 = make_dropout_layer(0.5, 0);
    Layer *l10 = make_connect_layer(4096, 1, "relu");
    // Layer *l11 = make_dropout_layer(0.5, 0);
    Layer *l12 = make_connect_layer(4096, 1, "relu");
    Layer *l13 = make_connect_layer(5, 1, "linear");
    Layer *l14 = make_softmax_layer(5);
    // Layer *l15 = make_mse_layer(5);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    // append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    // append_layer2grpah(g, l11);
    append_layer2grpah(g, l12);
    append_layer2grpah(g, l13);
    append_layer2grpah(g, l14);
    // append_layer2grpah(g, l15);
    init_kaiming_normal(l1, 0, "fan_in", "relu");
    init_kaiming_normal(l3, 0, "fan_in", "relu");
    init_kaiming_normal(l5, 0, "fan_in", "relu");
    init_kaiming_normal(l6, 0, "fan_in", "relu");
    init_kaiming_normal(l7, 0, "fan_in", "relu");

    init_kaiming_uniform(l10, 0, "fan_in", "relu");
    init_kaiming_uniform(l12, 0, "fan_in", "relu");
    init_kaiming_uniform(l13, 0, "fan_in", "linear");
    Session *sess = create_session(g, 224, 224, 3, 5, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    set_train_params(sess, 1, 1, 1, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/flower/train_c.txt", "./data/flower/train_c_label.txt");
    train(sess);
}

void alexnet_flower_detect(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(96, 11, 4, 2, 1, 0, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 0);
    Layer *l3 = make_convolutional_layer(256, 5, 1, 2, 1, 0, "relu");
    Layer *l4 = make_maxpool_layer(3, 2, 0);
    Layer *l5 = make_convolutional_layer(384, 3, 1, 1, 1, 0, "relu");
    Layer *l6 = make_convolutional_layer(384, 3, 1, 1, 1, 0, "relu");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, 0, "relu");
    Layer *l8 = make_global_maxpool_layer();
    Layer *l9 = make_connect_layer(128, 1, "relu");
    Layer *l10 = make_dropout_layer(0.5, 0);
    Layer *l11 = make_connect_layer(128, 1, "relu");
    Layer *l12 = make_dropout_layer(0.5, 0);
    Layer *l13 = make_connect_layer(5, 1, "linear");
    Layer *l14 = make_softmax_layer(5);
    Layer *l15 = make_mse_layer(5);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    append_layer2grpah(g, l11);
    append_layer2grpah(g, l12);
    append_layer2grpah(g, l13);
    append_layer2grpah(g, l14);
    append_layer2grpah(g, l15);
    Session *sess = create_session(g, 224, 224, 3, 5, type, path);
    // float *mean = calloc(3, sizeof(float));
    // float *std = calloc(3, sizeof(float));
    // mean[0] = 0.466;
    // mean[1] = 0.425;
    // mean[2] = 0.304;
    // std[0] = 0.299;
    // std[1] = 0.269;
    // std[2] = 0.293;
    // transform_normalize_sess(sess, mean, std);
    set_detect_params(sess);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    detect_classification(sess);
}

