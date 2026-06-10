#include "deeplabv1.h"

void deeplabv1(char *type, char *path)
{
    int num_class = 21;
    Graph *graph = create_graph();
    Layer **layers = malloc(23*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 3, 1, 1, 0, 1, "relu");
    layers[1] = make_convolutional_layer(64, 3, 1, 1, 0, 1, "relu");
    layers[2] = make_maxpool_layer(2, 2, 0);

    layers[3] = make_convolutional_layer(128, 3, 1, 1, 0, 1, "relu");
    layers[4] = make_convolutional_layer(128, 3, 1, 1, 0, 1, "relu");
    layers[5] = make_maxpool_layer(2, 2, 0);

    layers[6] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[7] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[8] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[9] = make_maxpool_layer(2, 2, 0);
    // pool3
    layers[10] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[11] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[12] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[13] = make_maxpool_layer(2, 2, 0);
    // pool4
    layers[14] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[15] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[17] = make_maxpool_layer(2, 2, 0);
    // fc6
    layers[18] = make_convolutional_layer(1024, 3, 1, 6, 5, 1, "relu");
    // fc7
    layers[19] = make_convolutional_layer(1024, 1, 1, 0, 0, 1, "relu");
    // score
    layers[20] = make_convolutional_layer(num_class, 1, 1, 0, 0, 1, "linear");
    // upsample
    layers[21] = make_interpolate_layer(320, 320);
    layers[22] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 23; ++i){
        append_layer2grpah(graph, layers[i]);
        Layer *l = layers[i];
        if (l->type == CONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, 0, "fan_in", "relu");
            init_constant_bias(l, 0);
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
    set_train_params(sess, 50, 8, 8, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 5e-4, 0, 0);
    init_session(sess, "./data/VOC2012/train.txt", "./data/VOC2012/train_label.txt");
    train(sess);
}

void deeplabv1_detect(char*type, char *path)
{
    int num_class = 21;
    Graph *graph = create_graph();
    Layer **layers = malloc(23*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 3, 1, 1, 0, 1, "relu");
    layers[1] = make_convolutional_layer(64, 3, 1, 1, 0, 1, "relu");
    layers[2] = make_maxpool_layer(2, 2, 0);

    layers[3] = make_convolutional_layer(128, 3, 1, 1, 0, 1, "relu");
    layers[4] = make_convolutional_layer(128, 3, 1, 1, 0, 1, "relu");
    layers[5] = make_maxpool_layer(2, 2, 0);

    layers[6] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[7] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[8] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[9] = make_maxpool_layer(2, 2, 0);
    // pool3
    layers[10] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[11] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[12] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[13] = make_maxpool_layer(2, 2, 0);
    // pool4
    layers[14] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[15] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[17] = make_maxpool_layer(2, 2, 0);
    // fc6
    layers[18] = make_convolutional_layer(1024, 3, 1, 6, 5, 1, "relu");
    // fc7
    layers[19] = make_convolutional_layer(1024, 1, 1, 0, 0, 1, "relu");
    // score
    layers[20] = make_convolutional_layer(num_class, 1, 1, 0, 0, 1, "linear");
    // upsample
    layers[21] = make_interpolate_layer(320, 320);
    layers[22] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 23; ++i){
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
