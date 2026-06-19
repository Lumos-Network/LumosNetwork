#include "yolov1.h"

void yolov1(char *type, char *path)
{
    int grid_size = 7;
    int num_bbox = 2;
    int num_classes = 20;
    Graph *graph = create_graph();
    Layer **layers = malloc(32*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 7, 2, 1, 0, 1, "relu");
    layers[1] = make_maxpool_layer(2, 2, 0);
    layers[2] = make_convolutional_layer(192, 3, 1, 1, 0, 1, "relu");
    layers[3] = make_maxpool_layer(2, 2, 0);
    layers[4] = make_convolutional_layer(128, 1, 1, 1, 0, 1, "relu");
    layers[5] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[6] = make_convolutional_layer(256, 1, 1, 1, 0, 1, "relu");
    layers[7] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[8] = make_maxpool_layer(2, 2, 0);

    layers[9] = make_convolutional_layer(256, 1, 1, 1, 0, 1, "relu");
    layers[10] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[11] = make_convolutional_layer(256, 1, 1, 1, 0, 1, "relu");
    layers[12] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[13] = make_convolutional_layer(256, 1, 1, 1, 0, 1, "relu");
    layers[14] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[15] = make_convolutional_layer(256, 1, 1, 1, 0, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");

    layers[17] = make_convolutional_layer(512, 1, 1, 1, 0, 1, "relu");
    layers[18] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[19] = make_maxpool_layer(2, 2, 0);

    layers[20] = make_convolutional_layer(512, 1, 1, 1, 0, 1, "relu");
    layers[21] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[22] = make_convolutional_layer(512, 1, 1, 1, 0, 1, "relu");
    layers[23] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[24] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[25] = make_convolutional_layer(1024, 3, 2, 1, 0, 1, "relu");
    layers[26] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");
    layers[27] = make_convolutional_layer(1024, 3, 1, 1, 0, 1, "relu");

    layers[28] = make_connect_layer(4096, 1, "leaky");
    layers[29] = make_dropout_layer(0.5);
    layers[30] = make_connect_layer(grid_size*grid_size*(num_bbox*5+num_classes), 1, "linear");
    layers[31] = make_yolo_layer();

    for (int i = 0; i < 32; ++i) {
        append_layer2grpah(graph, layers[i]);
        Layer *l = layers[i];
        if (l->type == CONNECT){
            init_kaiming_normal_kernel(l, sqrt(5.0), "fan_in", "relu");
            init_constant_bias(l, 0);
        }
    }

    Session *sess = create_session(graph, 448, 448, 3, grid_size*grid_size*(num_bbox*5+num_classes), num_classes, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 448, 448);
    set_train_params(sess, 200, 64, 64, 0.0005);
    SGDOptimizer_sess(sess, 0.9, 0, 5e-4, 0, 0);
    lr_scheduler_step(sess, 100, 0.1);
    init_session(sess, "./data/VOC2012/train_object.txt", "./data/VOC2012/train_object_label.txt");
    train(sess);
}

void yolov1_detect(char*type, char *path)
{

}
