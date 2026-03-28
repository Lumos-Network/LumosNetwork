#include "resnet18.h"

void resnet18(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(64, 7, 2, 3, 1, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 1);

    Layer *l3 = make_convolutional_layer(64, 3, 1, 1, 1, "relu");
    Layer *l4 = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    Layer *l5 = make_shortcut_layer(l3, 0, "relu");

    Layer *l6 = make_convolutional_layer(64, 3, 1, 1, 1, "relu");
    Layer *l7 = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    Layer *l8 = make_shortcut_layer(l6, 0, "relu");

    Layer *l9 = make_convolutional_layer(128, 3, 2, 1, 1, "relu");
    Layer *l10 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *l11 = make_shortcut_layer(l9, 1, "linear");
    Layer *l12 = make_convolutional_layer(128, 1, 2, 0, 1, "linear");
    Layer *l13 = make_shortcut_layer(l11, 0, "relu");

    Layer *l14 = make_convolutional_layer(128, 3, 1, 1, 1, "relu");
    Layer *l15 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *l16 = make_shortcut_layer(l14, 0, "relu");

    Layer *l17 = make_convolutional_layer(256, 3, 2, 1, 1, "relu");
    Layer *l18 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *l19 = make_shortcut_layer(l17, 1, "linear");
    Layer *l20 = make_convolutional_layer(256, 1, 2, 0, 1, "linear");
    Layer *l21 = make_shortcut_layer(l19, 0, "relu");

    Layer *l22 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l23 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *l24 = make_shortcut_layer(l22, 0, "relu");

    Layer *l25 = make_convolutional_layer(512, 3, 2, 1, 1, "relu");
    Layer *l26 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *l27 = make_shortcut_layer(l25, 1, "linear");
    Layer *l28 = make_convolutional_layer(512, 1, 2, 0, 1, "linear");
    Layer *l29 = make_shortcut_layer(l27, 0, "relu");

    Layer *l30 = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    Layer *l31 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *l32 = make_shortcut_layer(l30, 0, "relu");

    Layer *l33 = make_avgpool_layer(7, 1, 0);
    Layer *l34 = make_connect_layer(5, 1, "linear");
    Layer *l35 = make_crossentropy_layer(5);

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
    append_layer2grpah(g, l30);
    append_layer2grpah(g, l31);
    append_layer2grpah(g, l32);
    append_layer2grpah(g, l33);
    append_layer2grpah(g, l34);
    append_layer2grpah(g, l35);

    init_kaiming_uniform_kernel(l1, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l3, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l4, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l6, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l7, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l9, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l10, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l12, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l14, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l15, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l17, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l18, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l20, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l22, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l23, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l25, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l26, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l28, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l30, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l31, sqrt(5.0), "fan_in", "leaky_relu");
    init_kaiming_uniform_kernel(l34, sqrt(5.0), "fan_in", "leaky_relu");

    init_kaiming_uniform_bias(l1, "fan_in");
    init_kaiming_uniform_bias(l3, "fan_in");
    init_kaiming_uniform_bias(l4, "fan_in");
    init_kaiming_uniform_bias(l6, "fan_in");
    init_kaiming_uniform_bias(l7, "fan_in");
    init_kaiming_uniform_bias(l9, "fan_in");
    init_kaiming_uniform_bias(l10, "fan_in");
    init_kaiming_uniform_bias(l12, "fan_in");
    init_kaiming_uniform_bias(l14, "fan_in");
    init_kaiming_uniform_bias(l15, "fan_in");
    init_kaiming_uniform_bias(l17, "fan_in");
    init_kaiming_uniform_bias(l18, "fan_in");
    init_kaiming_uniform_bias(l20, "fan_in");
    init_kaiming_uniform_bias(l22, "fan_in");
    init_kaiming_uniform_bias(l23, "fan_in");
    init_kaiming_uniform_bias(l25, "fan_in");
    init_kaiming_uniform_bias(l26, "fan_in");
    init_kaiming_uniform_bias(l28, "fan_in");
    init_kaiming_uniform_bias(l30, "fan_in");
    init_kaiming_uniform_bias(l31, "fan_in");
    init_kaiming_uniform_bias(l34, "fan_in");

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
    set_train_params(sess, 20, 32, 32, 0.01);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    train(sess);
}

void resnet18_detect(char*type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(64, 7, 2, 3, 1, "relu");
    Layer *l2 = make_maxpool_layer(3, 2, 1);

    Layer *l3 = make_convolutional_layer(64, 3, 1, 1, 1, "relu");
    Layer *l4 = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    Layer *l5 = make_shortcut_layer(l3, 0, "relu");

    Layer *l6 = make_convolutional_layer(64, 3, 1, 1, 1, "relu");
    Layer *l7 = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    Layer *l8 = make_shortcut_layer(l6, 0, "relu");

    Layer *l9 = make_convolutional_layer(128, 3, 2, 1, 1, "relu");
    Layer *l10 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *l11 = make_shortcut_layer(l9, 1, "linear");
    Layer *l12 = make_convolutional_layer(128, 1, 2, 0, 1, "linear");
    Layer *l13 = make_shortcut_layer(l11, 0, "relu");

    Layer *l14 = make_convolutional_layer(128, 3, 1, 1, 1, "relu");
    Layer *l15 = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    Layer *l16 = make_shortcut_layer(l14, 0, "relu");

    Layer *l17 = make_convolutional_layer(256, 3, 2, 1, 1, "relu");
    Layer *l18 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *l19 = make_shortcut_layer(l17, 1, "linear");
    Layer *l20 = make_convolutional_layer(256, 1, 2, 0, 1, "linear");
    Layer *l21 = make_shortcut_layer(l19, 0, "relu");

    Layer *l22 = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    Layer *l23 = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    Layer *l24 = make_shortcut_layer(l22, 0, "relu");

    Layer *l25 = make_convolutional_layer(512, 3, 2, 1, 1, "relu");
    Layer *l26 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *l27 = make_shortcut_layer(l25, 1, "linear");
    Layer *l28 = make_convolutional_layer(512, 1, 2, 0, 1, "linear");
    Layer *l29 = make_shortcut_layer(l27, 0, "relu");

    Layer *l30 = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    Layer *l31 = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    Layer *l32 = make_shortcut_layer(l30, 0, "relu");

    Layer *l33 = make_avgpool_layer(7, 1, 0);
    Layer *l34 = make_connect_layer(5, 1, "linear");
    Layer *l35 = make_crossentropy_layer(5);

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
    append_layer2grpah(g, l30);
    append_layer2grpah(g, l31);
    append_layer2grpah(g, l32);
    append_layer2grpah(g, l33);
    append_layer2grpah(g, l34);
    append_layer2grpah(g, l35);
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
    set_detect_params(sess);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    detect_classification(sess);
}
