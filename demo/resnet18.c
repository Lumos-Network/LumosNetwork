#include "resnet18.h"

void resnet18(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(64, 7, 2, 3, 1, "leaky");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(64, 3, 1, 1, 1, "leaky");
    Layer *l4 = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    Layer *l5 = make_shortcut_layer(l2, "leaky");
    Layer *l6 = make_convolutional_layer(64, 3, 1, 1, 1, "leaky");
    Layer *l7 = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    Layer *l8 = make_shortcut_layer(l5, "leaky");
    Layer *l9 = make_convolutional_layer(128, 3, 2, 1, 1, "leaky");
    Layer *l10 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *l11 = make_shortcut_layer(l8, "leaky");
    Layer *l12 = make_convolutional_layer(128, 3, 1, 1, 1, "leaky");
    Layer *l13 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *l14 = make_shortcut_layer(l11, "leaky");
    Layer *l15 = make_convolutional_layer(256, 3, 2, 1, 1, "leaky");
    Layer *l16 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *l17 = make_shortcut_layer(l14, "leaky");
    Layer *l18 = make_convolutional_layer(256, 3, 1, 1, 1, "leaky");
    Layer *l19 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *l20 = make_shortcut_layer(l17, "leaky");
    Layer *l21 = make_convolutional_layer(512, 3, 2, 1, 1, "leaky");
    Layer *l22 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *l23 = make_shortcut_layer(l20, "leaky");
    Layer *l24 = make_convolutional_layer(512, 3, 1, 1, 1, "leaky");
    Layer *l25 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *l26 = make_shortcut_layer(l23, "leaky");
    Layer *l27 = make_global_avgpool_layer();
    Layer *l28 = make_connect_layer(5, 1, "linear");
    Layer *l29 = make_crossentropy_layer(5);

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
    append_layer2grpah(g, l18);
    append_layer2grpah(g, l19);
    append_layer2grpah(g, l20);
    append_layer2grpah(g, l21);
    append_layer2grpah(g, l22);
    append_layer2grpah(g, l23);
    append_layer2grpah(g, l24);
    append_layer2grpah(g, l25);
    append_layer2grpah(g, l26);
    append_layer2grpah(g, l27);
    append_layer2grpah(g, l28);
    append_layer2grpah(g, l29);

    init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l3, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l4, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l6, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l7, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l9, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l10, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l12, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l13, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l15, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l16, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l18, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l19, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l21, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l22, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l24, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l25, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_kernel(l28, sqrt(5.0), "fan_in", "leaky_relu");

    init_constant_bias(l1, 0);
    init_constant_bias(l3, 0);
    init_constant_bias(l4, 0);
    init_constant_bias(l6, 0);
    init_constant_bias(l7, 0);
    init_constant_bias(l9, 0);
    init_constant_bias(l10, 0);
    init_constant_bias(l12, 0);
    init_constant_bias(l13, 0);
    init_constant_bias(l15, 0);
    init_constant_bias(l16, 0);
    init_constant_bias(l18, 0);
    init_constant_bias(l19, 0);
    init_constant_bias(l21, 0);
    init_constant_bias(l22, 0);
    init_constant_bias(l24, 0);
    init_constant_bias(l25, 0);
    init_constant_bias(l28, 0);

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
    transform_resize_sess(sess, 224, 224);
    set_train_params(sess, 20, 32, 32, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    train(sess);
}

void resnet18_detect(char*type, char *path)
{
    
}
