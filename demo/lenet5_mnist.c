#include "lenet5_mnist.h"

void lenet5_mnist(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, "relu");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, "relu");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, "relu");
    Layer *l6 = make_connect_layer(84, 1, "relu");
    Layer *l7 = make_connect_layer(10, 1, "linear");
    Layer *l8 = make_crossentropy_layer(10);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);

    init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu"); //默认使用leaky relu
    init_kaiming_uniform_kernel(l3, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l5, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_bias(l1, "fan_in");
    init_kaiming_uniform_bias(l3, "fan_in");
    init_kaiming_uniform_bias(l5, "fan_in");

    init_kaiming_uniform_kernel(l6, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l7, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_bias(l6, "fan_in");
    init_kaiming_uniform_bias(l7, "fan_in");

    Session *sess = create_session(g, 32, 32, 1, 10, type, path);
    set_train_params(sess, 10, 16, 16, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/mnist/train.txt", "./data/mnist/train_label.txt");
    train(sess);
}

void lenet5_mnist_detect(char*type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, "relu");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, "relu");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, "relu");
    Layer *l6 = make_connect_layer(84, 1, "relu");
    Layer *l7 = make_connect_layer(10, 1, "linear");
    Layer *l8 = make_crossentropy_layer(10);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    Session *sess = create_session(g, 32, 32, 1, 10, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/mnist/test.txt", "./data/mnist/test_label.txt");
    detect_classification(sess);
}
