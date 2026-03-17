#include "lenet5_fmnist.h"

void lenet5_fmnist(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 2, 1, "relu");
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

    init_kaiming_uniform(l1, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform(l3, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform(l5, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform(l6, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform(l7, sqrt(5.0), "fan_in", "leaky_relu");

    Session *sess = create_session(g, 28, 28, 1, 10, type, path);
    set_train_params(sess, 20, 32, 32, 0.01);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/fmnist/train.txt", "./data/fmnist/label.txt");
    train(sess);
}

void lenet5_fmnist_detect(char*type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 2, 1, "relu");
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
    Session *sess = create_session(g, 28, 28, 1, 10, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/fmnist/train.txt", "./data/fmnist/label.txt");
    detect_classification(sess);
}
