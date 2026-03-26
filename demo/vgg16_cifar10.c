#include "vgg16_cifar10.h"

void vgg16_cifar10(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(64, 3, 1, 1, 1, "relu");
    Layer *l3 = make_convolutional_layer(64, 3, 1, 1, 1, "relu");
    Layer *l5 = make_maxpool_layer(2, 2, 0);

    Layer *l6 = make_convolutional_layer(128, 3, 1, 1, 1, "relu");
    Layer *l8 = make_convolutional_layer(128, 3, 1, 1, 1, "relu");
    Layer *l10 = make_maxpool_layer(2, 2, 0);

    Layer *l11 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l13 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l15 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l17 = make_maxpool_layer(2, 2, 0);

    Layer *l18 = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    Layer *l20 = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    Layer *l22 = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    Layer *l24 = make_maxpool_layer(2, 2, 0);

    Layer *l25 = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    Layer *l27 = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    Layer *l29 = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    Layer *l31 = make_maxpool_layer(2, 2, 0);

    Layer *l32 = make_dropout_layer(0.5);
    Layer *l33 = make_connect_layer(4096, 1, "relu");
    Layer *l34 = make_dropout_layer(0.5);
    Layer *l35 = make_connect_layer(4096, 1, "relu");
    Layer *l36 = make_connect_layer(4, 1, "linear");
    Layer *l37 = make_crossentropy_layer(4);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l10);
    append_layer2grpah(g, l11);
    append_layer2grpah(g, l13);
    append_layer2grpah(g, l15);
    append_layer2grpah(g, l17);
    append_layer2grpah(g, l18);
    append_layer2grpah(g, l20);
    append_layer2grpah(g, l22);
    append_layer2grpah(g, l24);
    append_layer2grpah(g, l25);
    append_layer2grpah(g, l27);
    append_layer2grpah(g, l29);
    append_layer2grpah(g, l31);
    append_layer2grpah(g, l32);
    append_layer2grpah(g, l33);
    append_layer2grpah(g, l34);
    append_layer2grpah(g, l35);
    append_layer2grpah(g, l36);
    append_layer2grpah(g, l37);

    init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l3, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l6, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l8, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l11, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l13, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l15, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l18, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l20, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l22, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l25, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l27, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l29, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l33, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l35, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l36, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_bias(l1, "fan_in");
    init_kaiming_uniform_bias(l3, "fan_in");
    init_kaiming_uniform_bias(l6, "fan_in");
    init_kaiming_uniform_bias(l8, "fan_in");
    init_kaiming_uniform_bias(l11, "fan_in");
    init_kaiming_uniform_bias(l13, "fan_in");
    init_kaiming_uniform_bias(l15, "fan_in");
    init_kaiming_uniform_bias(l18, "fan_in");
    init_kaiming_uniform_bias(l20, "fan_in");
    init_kaiming_uniform_bias(l22, "fan_in");
    init_kaiming_uniform_bias(l25, "fan_in");
    init_kaiming_uniform_bias(l27, "fan_in");
    init_kaiming_uniform_bias(l29, "fan_in");

    init_kaiming_uniform_bias(l33, "fan_in");
    init_kaiming_uniform_bias(l35, "fan_in");
    init_kaiming_uniform_bias(l36, "fan_in");

    Session *sess = create_session(g, 224, 224, 3, 4, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 224, 224);
    set_train_params(sess, 40, 32, 32, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/coffee/train.txt", "./data/coffee/train_label.txt");
    train(sess);
}
