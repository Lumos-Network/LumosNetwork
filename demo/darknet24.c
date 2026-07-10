#include "darknet24.h"

void darknet24(char *type, char *path)
{
    int num_classes = 20;
    Graph *graph = create_graph();
    Layer **layers = malloc(31*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 7, 2, 3, 0, 1, "relu");
    layers[1] = make_maxpool_layer(2, 2, 0);
    layers[2] = make_convolutional_layer(192, 3, 1, 1, 0, 1, "relu");
    layers[3] = make_maxpool_layer(2, 2, 0);
    layers[4] = make_convolutional_layer(128, 1, 1, 0, 0, 1, "relu");
    layers[5] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[6] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[7] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[8] = make_maxpool_layer(2, 2, 0);

    layers[9] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[10] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[11] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[12] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[13] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[14] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[15] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");

    layers[17] = make_convolutional_layer(512, 1, 1, 0, 0, 1, "relu");
    layers[18] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[19] = make_maxpool_layer(2, 2, 0);

    layers[20] = make_convolutional_layer(512, 1, 1, 0, 0, 1, "relu");
    layers[21] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[22] = make_convolutional_layer(512, 1, 1, 0, 0, 1, "relu");
    layers[23] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    // 预训练分割
    layers[24] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[25] = make_convolutional_layer(1024, 3, 2, 1, 0, 1, "relu");
    layers[26] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[27] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");

    layers[28] = make_avgpool_layer(7, 7, 0);
    layers[29] = make_connect_layer(num_classes, 1, "linear");
    layers[30] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 31; ++i) {
        append_layer2grpah(graph, layers[i]);
        Layer *l = layers[i];
        if (l->type == CONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, 0, "fan_out", "relu");
            init_constant_bias(l, 0);
        }
        if (l->type == CONNECT){
            init_kaiming_normal_kernel(l, 0, "fan_out", "relu");
            init_constant_bias(l, 0);
        }
    }

    Session *sess = create_session(graph, 224, 224, 3, 1, num_classes, type, path);
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
    set_train_params(sess, 50, 64, 64, 0.00001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/VOC2012/train_classifier.txt", "./data/VOC2012/train_classifier_label.txt");
    train(sess);
}

void darknet24_detect(char*type, char *path)
{
    int num_classes = 20;
    Graph *graph = create_graph();
    Layer **layers = malloc(31*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 7, 2, 3, 0, 1, "relu");
    layers[1] = make_maxpool_layer(2, 2, 0);
    layers[2] = make_convolutional_layer(192, 3, 1, 1, 0, 1, "relu");
    layers[3] = make_maxpool_layer(2, 2, 0);
    layers[4] = make_convolutional_layer(128, 1, 1, 0, 0, 1, "relu");
    layers[5] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[6] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[7] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[8] = make_maxpool_layer(2, 2, 0);

    layers[9] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[10] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[11] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[12] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[13] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[14] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[15] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");

    layers[17] = make_convolutional_layer(512, 1, 1, 0, 0, 1, "relu");
    layers[18] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[19] = make_maxpool_layer(2, 2, 0);

    layers[20] = make_convolutional_layer(512, 1, 1, 0, 0, 1, "relu");
    layers[21] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[22] = make_convolutional_layer(512, 1, 1, 0, 0, 1, "relu");
    layers[23] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    // 预训练分割
    layers[24] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[25] = make_convolutional_layer(1024, 3, 2, 1, 0, 1, "relu");
    layers[26] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[27] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");

    layers[28] = make_avgpool_layer(7, 7, 0);
    layers[29] = make_connect_layer(num_classes, 1, "linear");
    layers[30] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 31; ++i) {
        append_layer2grpah(graph, layers[i]);
    }

    Session *sess = create_session(graph, 224, 224, 3, 1, num_classes, type, path);
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
    init_session(sess, "./data/VOC2012/train_classifier.txt", "./data/VOC2012/train_classifier_label.txt");
    detect_classification(sess);
}
