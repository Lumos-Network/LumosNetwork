#include "yolov1.h"

void yolov1(char *type, char *path)
{
    int grid_size = 7;
    int num_bbox = 2;
    int num_classes = 20;
    Graph *graph = create_graph();
    Layer **layers = malloc(51*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 7, 2, 1, 0, 0, "linear");
    layers[1] = make_normalization_layer(0.1, 1, "leaky");
    layers[2] = make_maxpool_layer(2, 2, 0);
    layers[3] = make_convolutional_layer(192, 3, 1, 1, 0, 0, "linear");
    layers[4] = make_normalization_layer(0.1, 1, "leaky");
    layers[5] = make_maxpool_layer(2, 2, 0);
    layers[6] = make_convolutional_layer(128, 1, 1, 1, 0, 0, "linear");
    layers[7] = make_normalization_layer(0.1, 1, "leaky");
    layers[8] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "linear");
    layers[9] = make_normalization_layer(0.1, 1, "leaky");
    layers[10] = make_convolutional_layer(256, 1, 1, 1, 0, 0, "linear");
    layers[11] = make_normalization_layer(0.1, 1, "leaky");
    layers[12] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[13] = make_normalization_layer(0.1, 1, "leaky");
    layers[14] = make_maxpool_layer(2, 2, 0);

    layers[15] = make_convolutional_layer(256, 1, 1, 1, 0, 0, "linear");
    layers[16] = make_normalization_layer(0.1, 1, "leaky");
    layers[17] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[18] = make_normalization_layer(0.1, 1, "leaky");
    layers[19] = make_convolutional_layer(256, 1, 1, 1, 0, 0, "linear");
    layers[20] = make_normalization_layer(0.1, 1, "leaky");
    layers[21] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[22] = make_normalization_layer(0.1, 1, "leaky");
    layers[23] = make_convolutional_layer(256, 1, 1, 1, 0, 0, "linear");
    layers[24] = make_normalization_layer(0.1, 1, "leaky");
    layers[25] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[26] = make_normalization_layer(0.1, 1, "leaky");
    layers[27] = make_convolutional_layer(256, 1, 1, 1, 0, 0, "linear");
    layers[28] = make_normalization_layer(0.1, 1, "leaky");
    layers[29] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[30] = make_normalization_layer(0.1, 1, "leaky");

    layers[31] = make_convolutional_layer(512, 1, 1, 1, 0, 0, "linear");
    layers[32] = make_normalization_layer(0.1, 1, "leaky");
    layers[33] = make_convolutional_layer(1024, 3, 1, 1, 0, 0, "linear");
    layers[34] = make_normalization_layer(0.1, 1, "leaky");
    layers[35] = make_maxpool_layer(2, 2, 0);

    layers[36] = make_convolutional_layer(512, 1, 1, 1, 0, 0, "linear");
    layers[37] = make_normalization_layer(0.1, 1, "leaky");
    layers[38] = make_convolutional_layer(1024, 3, 1, 1, 0, 0, "linear");
    layers[39] = make_normalization_layer(0.1, 1, "leaky");
    layers[40] = make_convolutional_layer(512, 1, 1, 1, 0, 0, "linear");
    layers[41] = make_normalization_layer(0.1, 1, "leaky");
    layers[42] = make_convolutional_layer(1024, 3, 1, 1, 0, 0, "linear");
    layers[43] = make_normalization_layer(0.1, 1, "leaky");
    layers[44] = make_convolutional_layer(1024, 3, 1, 1, 0, 0, "linear");
    layers[45] = make_normalization_layer(0.1, 1, "leaky");
    layers[46] = make_convolutional_layer(1024, 3, 2, 1, 0, 0, "linear");
    layers[47] = make_normalization_layer(0.1, 1, "leaky");
    layers[48] = make_convolutional_layer(1024, 3, 1, 1, 0, 0, "linear");
    layers[49] = make_normalization_layer(0.1, 1, "leaky");
    layers[50] = make_convolutional_layer(1024, 3, 1, 1, 0, 0, "linear");
    layers[51] = make_normalization_layer(0.1, 1, "leaky");

    layers[48] = make_connect_layer(4096, 1, "leaky");
    layers[49] = make_dropout_layer(0.5);
    layers[50] = make_connect_layer(grid_size*grid_size*(num_bbox*5+num_classes), 1, "linear");

    for (int i = 0; i < 51; ++i) {
        append_layer2grpah(graph, layers[i]);
    }

    Session *sess = create_session(graph, 224, 224, 3, 1, 1000, type, path);
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
    set_train_params(sess, 300, 64, 64, 0.01);
    SGDOptimizer_sess(sess, 0.9, 0, 5e-4, 0, 0);
    lr_scheduler_step(sess, 100, 0.1);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    train(sess);
}

void yolov1_detect(char*type, char *path)
{

}
