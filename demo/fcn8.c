#include "fcn8.h"

void fcn8(char *type, char *path)
{
    Graph *graph = create_graph();
    Layer **layers = malloc(30*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 3, 1, 1, 1, "relu");
    layers[1] = make_convolutional_layer(64, 3, 1, 1, 1, "relu");
    layers[2] = make_maxpool_layer(2, 2, 0);

    layers[3] = make_convolutional_layer(128, 3, 1, 1, 1, "relu");
    layers[4] = make_convolutional_layer(128, 3, 1, 1, 1, "relu");
    layers[5] = make_maxpool_layer(2, 2, 0);

    layers[6] = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    layers[7] = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    layers[8] = make_convolutional_layer(256, 3, 1, 1, 1, "relu");
    layers[9] = make_maxpool_layer(2, 2, 0);
    // pool3
    layers[10] = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    layers[11] = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    layers[12] = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    layers[13] = make_maxpool_layer(2, 2, 0);
    // pool4
    layers[14] = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    layers[15] = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 1, 1, "relu");
    layers[17] = make_maxpool_layer(2, 2, 0);
    // pool5
    layers[18] = make_convolutional_layer(4096, 7, 1, 3, 1, "linear");
    layers[19] = make_convolutional_layer(4096, 1, 1, 0, 1, "linear");
    layers[20] = make_convolutional_layer(21, 1, 1, 0, 1, "linear");
    // fc8
    layers[21] = make_deconvolutional_layer(21, 4, 2, 1, 1, "linear");
    layers[22] = make_shortcut_layer(layers[14], 1, "linear");
    layers[23] = make_convolutional_layer(21, 1, 1, 0, 1, "linear");
    Layer **fuse_pool4 = malloc(2*sizeof(Layer*));
    layers[24] = make_inception_layer(fuse_pool4, 2, 2);
    fuse_pool4[0] = layers[22];
    fuse_pool4[1] = layers[24];
    // fuse pool4
    layers[25] = make_deconvolutional_layer(21, 4, 2, 1, 1, "linear");
    layers[26] = make_shortcut_layer(layers[10], 1, "linear");
    Layer **fuse_pool3 = malloc(2*sizeof(Layer*));
    layers[27] = make_inception_layer(fuse_pool3, 2, 2);
    fuse_pool3[0] = layers[26];
    fuse_pool3[1] = layers[27];

    layers[28] = make_deconvolutional_layer(21, 16, 8, 4, 1, "linear");
    layers[29] = make_crossentropy_layer(NULL, 255);

    for (int i = 0; i < 30; ++i){
        append_layer2grpah(graph, layers[i]);
        Layer *l = layers[i];
        if (l->type == CONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, sqrt(5.0), "fan_in", "relu");
            init_constant_bias(l, 0);
        }
        if (l->type == DECONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, sqrt(5.0), "fan_in", "relu");
            init_constant_bias(l, 0);
        }
    }
    Session *sess = create_session(graph, 320, 320, 3, 320*320, 21, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    set_train_params(sess, 30, 8, 8, 0.0001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/VOC2012/train.txt", "./data/VOC2012/train_label.txt");
    train(sess);
}
