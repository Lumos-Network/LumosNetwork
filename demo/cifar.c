#include "cifar.h"

void cifar(char *type, char *path)
{
    Graph *graph = create_graph();
    Layer *l1 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *b1 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l2 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *b2 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l3 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *b3 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_dropout_layer(0.5);
    Layer *l6 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *b4 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *b5 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l8 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *b6 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l9 = make_maxpool_layer(2, 2, 0);
    Layer *l10 = make_dropout_layer(0.5);
    Layer *l11 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *b7 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l12 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *b8 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l13 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *b9 = make_normalization_layer(0.1, 1, "leaky");
    Layer *l14 = make_dropout_layer(0.5);
    Layer *l15 = make_convolutional_layer(10, 1, 1, 1, 1, "leaky");
    Layer *l16 = make_avgpool_layer(9, 9, 0);
    Layer *l17 = make_crossentropy_layer(10);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, b1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, b2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, b3);
    append_layer2grpah(graph, l4);
    append_layer2grpah(graph, l5);
    append_layer2grpah(graph, l6);
    append_layer2grpah(graph, b4);
    append_layer2grpah(graph, l7);
    append_layer2grpah(graph, b5);
    append_layer2grpah(graph, l8);
    append_layer2grpah(graph, b6);
    append_layer2grpah(graph, l9);
    append_layer2grpah(graph, l10);
    append_layer2grpah(graph, l11);
    append_layer2grpah(graph, b7);
    append_layer2grpah(graph, l12);
    append_layer2grpah(graph, b8);
    append_layer2grpah(graph, l13);
    append_layer2grpah(graph, b9);
    append_layer2grpah(graph, l14);
    append_layer2grpah(graph, l15);
    append_layer2grpah(graph, l16);
    append_layer2grpah(graph, l17);

    init_kaiming_uniform_kernel(l1, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l2, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l3, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l6, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l7, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l8, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l11, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l12, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l13, sqrt(5), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l15, sqrt(5), "fan_in", "leaky_relu");

    init_constant_bias(l1, 0);
    init_constant_bias(l2, 0);
    init_constant_bias(l3, 0);
    init_constant_bias(l6, 0);
    init_constant_bias(l7, 0);
    init_constant_bias(l8, 0);
    init_constant_bias(l11, 0);
    init_constant_bias(l12, 0);
    init_constant_bias(l13, 0);
    init_constant_bias(l15, 0);

    Session *sess = create_session(graph, 28, 28, 3, 10, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.4914;
    mean[1] = 0.4822;
    mean[2] = 0.4465;
    std[0] = 0.247;
    std[1] = 0.243;
    std[2] = 0.262;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 28, 28);
    set_train_params(sess, 50, 32, 32, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/cifar10/train.txt", "./data/cifar10/train_label.txt");
    train(sess);
}

void cifar_detect(char*type, char *path)
{

}
