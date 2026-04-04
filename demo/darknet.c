#include "darknet.h"

void darknet(char *type, char *path)
{
    Graph *graph = create_graph();
    Layer *l1 = make_convolutional_layer(16, 3, 1, 1, 1, "linear");
    Layer *l2 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l3 = make_maxpool_layer(2, 2, 0);
    Layer *l4 = make_convolutional_layer(32, 3, 1, 1, 1, "linear");
    Layer *l5 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l6 = make_maxpool_layer(2, 2, 0);
    Layer *l7 = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    Layer *l8 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l9 = make_maxpool_layer(2, 2, 0);
    Layer *l10 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *l11 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l12 = make_maxpool_layer(2, 2, 0);
    Layer *l13 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *l14 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l15 = make_maxpool_layer(2, 2, 0);
    Layer *l16 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *l17 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l18 = make_maxpool_layer(2, 2, 0);
    Layer *l19 = make_convolutional_layer(1024, 3, 1, 1, 1, "linear");
    Layer *l20 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l21 = make_avgpool_layer(4, 4, 0);
    Layer *l22 = make_convolutional_layer(5, 1, 1, 0, 1, "linear");
    Layer *l23 = make_crossentropy_layer(5);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);
    append_layer2grpah(graph, l5);
    append_layer2grpah(graph, l6);
    append_layer2grpah(graph, l7);
    append_layer2grpah(graph, l8);
    append_layer2grpah(graph, l9);
    append_layer2grpah(graph, l10);
    append_layer2grpah(graph, l11);
    append_layer2grpah(graph, l12);
    append_layer2grpah(graph, l13);
    append_layer2grpah(graph, l14);
    append_layer2grpah(graph, l15);
    append_layer2grpah(graph, l16);
    append_layer2grpah(graph, l17);
    append_layer2grpah(graph, l18);
    append_layer2grpah(graph, l19);
    append_layer2grpah(graph, l20);
    append_layer2grpah(graph, l21);
    append_layer2grpah(graph, l22);
    append_layer2grpah(graph, l23);
    init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l4, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l7, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l10, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l13, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l16, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l19, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l22, sqrt(5.0), "fan_in", "leaky_relu");
    init_constant_bias(l1, 0);
    init_constant_bias(l4, 0);
    init_constant_bias(l7, 0);
    init_constant_bias(l10, 0);
    init_constant_bias(l13, 0);
    init_constant_bias(l16, 0);
    init_constant_bias(l19, 0);
    init_constant_bias(l22, 0);
    Session *sess = create_session(graph, 256, 256, 3, 5, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 256, 256);
    set_train_params(sess, 50, 32, 32, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    train(sess);
}

void darknet_detect(char*type, char *path)
{

}
