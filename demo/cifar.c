#include "cifar.h"

void cifar(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(128, 3, 1, 1, 1, "leaky");
    Layer *l2 = make_convolutional_layer(128, 3, 1, 1, 1, "leaky");
    Layer *l3 = make_convolutional_layer(128, 3, 1, 1, 1, "leaky");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_dropout_layer(0.5);

    Layer *l6 = make_convolutional_layer(256, 3, 1, 1, 1, "leaky");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, "leaky");
    Layer *l8 = make_convolutional_layer(256, 3, 1, 1, 1, "leaky");
    Layer *l9 = make_maxpool_layer(2, 2, 0);
    Layer *l10 = make_dropout_layer(0.5);

    Layer *l11 = make_convolutional_layer(512, 3, 1, 1, 1, "leaky");
    Layer *l12 = make_convolutional_layer(512, 3, 1, 1, 1, "leaky");
    Layer *l13 = make_convolutional_layer(512, 3, 1, 1, 1, "leaky");
    Layer *l14 = make_dropout_layer(0.5);

    Layer *l15 = make_convolutional_layer(10, 1, 1, 1, 1, "leaky");
    Layer *l16 = make_global_avgpool_layer();
    Layer *l17 = make_crossentropy_layer(10);

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
    append_layer2grpah(g, l16);
    append_layer2grpah(g, l17);

    init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l2, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l3, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l6, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l7, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l8, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l11, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l12, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l13, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l15, sqrt(5.0), "fan_in", "leaky_relu");

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

    Session *sess = create_session(g, 28, 28, 3, 10, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.4914;
    mean[1] = 0.4822;
    mean[2] = 0.4465;
    std[0] = 0.2023;
    std[1] = 0.1994;
    std[2] = 0.2010;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 28, 28);
    set_train_params(sess, 20, 128, 128, 0.0001);
    SGDOptimizer_sess(sess, 0, 0, 0, 0, 0);
    init_session(sess, "./data/cifar10/train.txt", "./data/cifar10/train_label.txt");
    train(sess);
}

void cifar_detect(char*type, char *path)
{

}
