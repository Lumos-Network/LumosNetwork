#include "xor.h"

void xor(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_connect_layer(8, 1, "relu");
    Layer *l2 = make_connect_layer(16, 1, "relu");
    Layer *l3 = make_connect_layer(2, 1, "linear");
    Layer *l4 = make_crossentropy_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l2, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l3, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_bias(l1, "fan_in");
    init_kaiming_uniform_bias(l2, "fan_in");
    init_kaiming_uniform_bias(l3, "fan_in");
    Session *sess = create_session(g, 1, 2, 1, 2, type, path);
    set_train_params(sess, 50, 4, 4, 0.1);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/xor/data.txt", "./data/xor/label.txt");
    train(sess);
}

void xor_detect(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_connect_layer(8, 1, "relu");
    Layer *l2 = make_connect_layer(16, 1, "relu");
    Layer *l3 = make_connect_layer(2, 1, "linear");
    Layer *l4 = make_crossentropy_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    Session *sess = create_session(g, 1, 2, 1, 2, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/xor/data.txt", "./data/xor/label.txt");
    detect_classification(sess);
}
