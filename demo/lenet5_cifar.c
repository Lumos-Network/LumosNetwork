#include "lenet5_cifar.h"

void lenet5_cifar(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(32, 3, 1, 3, 1, 0, "relu");
    Layer *l2 = make_convolutional_layer(48, 3, 1, 3, 1, 0, "relu");
    Layer *l3 = make_maxpool_layer(2, 2, 0);
    Layer *l4 = make_dropout_layer(0.25);
    Layer *l5 = make_convolutional_layer(80, 3, 1, 3, 1, 0, "relu");
    Layer *l6 = make_maxpool_layer(2, 2, 0);
    Layer *l7 = make_dropout_layer(0.25);
    Layer *l8 = make_convolutional_layer(128, 3, 1, 3, 1, 0, "relu");
    Layer *l9 = make_dropout_layer(0.25);
    Layer *l10 = make_connect_layer(500, 1, "relu");
    Layer *l11 = make_dropout_layer(0.25);
    Layer *l12 = make_connect_layer(10, 1, "relu");
    Layer *l13 = make_softmax_layer(10);
    Layer *l14 = make_mse_layer(10);
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

    init_kaiming_normal(l1, 0, "fan_in", "relu");
    init_kaiming_normal(l2, 0, "fan_in", "relu");
    init_kaiming_normal(l5, 0, "fan_in", "relu");
    init_kaiming_normal(l8, 0, "fan_in", "relu");

    init_kaiming_uniform(l10, 0, "fan_in", "relu");
    init_kaiming_uniform(l12, 0, "fan_in", "relu");

    Session *sess = create_session(g, 32, 32, 3, 10, type, path);
    set_train_params(sess, 50, 16, 16, 0.001);
    init_session(sess, "./data/cifar10/train.txt", "./data/cifar10/train_label.txt");
    train(sess, 0);
}

void lenet5_cifar_detect(char*type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(32, 3, 1, 3, 1, 0, "relu");
    Layer *l2 = make_convolutional_layer(48, 3, 1, 3, 1, 0, "relu");
    Layer *l3 = make_maxpool_layer(2, 2, 0);
    Layer *l4 = make_dropout_layer(0.25);
    Layer *l5 = make_convolutional_layer(80, 3, 1, 3, 1, 0, "relu");
    Layer *l6 = make_maxpool_layer(2, 2, 0);
    Layer *l7 = make_dropout_layer(0.25);
    Layer *l8 = make_convolutional_layer(128, 3, 1, 3, 1, 0, "relu");
    Layer *l9 = make_dropout_layer(0.25);
    Layer *l10 = make_connect_layer(500, 1, "relu");
    Layer *l11 = make_dropout_layer(0.25);
    Layer *l12 = make_connect_layer(10, 1, "relu");
    Layer *l13 = make_softmax_layer(10);
    Layer *l14 = make_mse_layer(10);
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
    Session *sess = create_session(g, 32, 32, 3, 10, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/cifar10/test.txt", "./data/cifar10/test_label.txt");
    detect_classification(sess, 0);
}
