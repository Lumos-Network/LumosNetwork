#include "fcn8.h"

void fcn8(char *type, char *path)
{
    int num_class = 21;
    Graph *graph = create_graph();
    Layer **layers = malloc(33*sizeof(Layer*));
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
    layers[18] = make_convolutional_layer(4096, 7, 1, 3, 1, "relu"); // fc6
    layers[19] = make_dropout_layer(0.5);
    layers[20] = make_convolutional_layer(4096, 1, 1, 0, 1, "relu"); // fc7
    layers[21] = make_dropout_layer(0.5);
    // 跳跃连接+上采样
    layers[22] = make_convolutional_layer(num_class, 1, 1, 0, 1, "linear"); // score_fr
    layers[23] = make_deconvolutional_layer(num_class, 4, 2, 1, 0, "linear"); // up2

    layers[24] = make_shortcut_layer(layers[14], 1, "linear");
    layers[25] = make_convolutional_layer(num_class, 1, 1, 0, 1, "linear"); // score_pool4
    layers[26] = make_shortcut_layer(layers[24], 0, "linear"); // fuse1

    layers[27] = make_deconvolutional_layer(num_class, 4, 2, 1, 0, "linear"); // up4

    layers[28] = make_shortcut_layer(layers[10], 1, "linear");
    layers[29] = make_convolutional_layer(num_class, 1, 1, 0, 1, "linear"); // score_pool3
    layers[30] = make_shortcut_layer(layers[28], 0, "linear");

    layers[31] = make_deconvolutional_layer(num_class, 16, 8, 4, 0, "linear");
    layers[32] = make_crossentropy_layer(NULL, 255);

    for (int i = 0; i < 33; ++i){
        append_layer2grpah(graph, layers[i]);
        Layer *l = layers[i];
        if (l->type == CONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, 0, "fan_in", "relu");
            init_constant_bias(l, 0);
        }
        if (l->type == DECONVOLUTIONAL){
            init_bilinearinterp_kernel(l);
        }
    }
    Session *sess = create_session(graph, 320, 320, 3, 320*320, num_class, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 320, 320);
    set_train_params(sess, 200, 20, 20, 1e-4);
    SGDOptimizer_sess(sess, 0.9, 0, 2e-4, 0, 0);
    init_session(sess, "./data/VOC2012/train.txt", "./data/VOC2012/train_label.txt");
    train(sess);
}

void fcn8_detect(char *type, char *path)
{
    int num_class = 21;
    Graph *graph = create_graph();
    Layer **layers = malloc(33*sizeof(Layer*));
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
    layers[18] = make_convolutional_layer(4096, 7, 1, 3, 1, "relu"); // fc6
    layers[19] = make_dropout_layer(0.5);
    layers[20] = make_convolutional_layer(4096, 1, 1, 0, 1, "relu"); // fc7
    layers[21] = make_dropout_layer(0.5);
    // 跳跃连接+上采样
    layers[22] = make_convolutional_layer(num_class, 1, 1, 0, 1, "linear"); // score_fr
    layers[23] = make_deconvolutional_layer(num_class, 4, 2, 1, 0, "linear"); // up2

    layers[24] = make_shortcut_layer(layers[14], 1, "linear");
    layers[25] = make_convolutional_layer(num_class, 1, 1, 0, 1, "linear"); // score_pool4
    layers[26] = make_shortcut_layer(layers[24], 0, "linear"); // fuse1

    layers[27] = make_deconvolutional_layer(num_class, 4, 2, 1, 0, "linear"); // up4

    layers[28] = make_shortcut_layer(layers[10], 1, "linear");
    layers[29] = make_convolutional_layer(num_class, 1, 1, 0, 1, "linear"); // score_pool3
    layers[30] = make_shortcut_layer(layers[28], 0, "linear");

    layers[31] = make_deconvolutional_layer(num_class, 16, 8, 4, 0, "linear");
    layers[32] = make_crossentropy_layer(NULL, 255);

    for (int i = 0; i < 33; ++i){
        append_layer2grpah(graph, layers[i]);
    }

    Session *sess = create_session(graph, 320, 320, 3, 320*320, num_class, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 320, 320);
    set_detect_params(sess);
    init_session(sess, "./data/VOC2012/train.txt", "./data/VOC2012/train_label.txt");
    detect_segmentation(sess);
}
