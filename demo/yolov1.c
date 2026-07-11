#include "yolov1.h"

void yolov1(char *type, char *path)
{
    int stride = 32;
    int num_bbox = 1;
    int num_classes = 20;
    int size = 416;
    int grids = size / stride;
    Graph *graph = create_graph();
    Layer **layers = malloc(70*sizeof(Layer*));
    // backbone-resnet18
    layers[0] = make_convolutional_layer(64, 7, 2, 3, 0, 0, "linear");
    layers[1] = make_normalization_layer(0.1, 1, "relu");
    layers[2] = make_maxpool_layer(3, 2, 1);

    layers[3] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "linear");
    layers[4] = make_normalization_layer(0.1, 1, "relu");
    layers[5] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "linear");
    layers[6] = make_normalization_layer(0.1, 1, "linear");
    layers[7] = make_shortcut_layer(layers[3], 0, "relu");

    layers[8] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "linear");
    layers[9] = make_normalization_layer(0.1, 1, "relu");
    layers[10] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "linear");
    layers[11] = make_normalization_layer(0.1, 1, "linear");
    layers[12] = make_shortcut_layer(layers[8], 0, "relu");

    layers[13] = make_convolutional_layer(128, 3, 2, 1, 0, 0, "linear");
    layers[14] = make_normalization_layer(0.1, 1, "relu");
    layers[15] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "linear");
    layers[16] = make_normalization_layer(0.1, 1, "linear");
    layers[17] = make_shortcut_layer(layers[13], 1, "linear");
    layers[18] = make_convolutional_layer(128, 1, 2, 0, 0, 0, "linear");
    layers[19] = make_normalization_layer(0.1, 1, "linear");
    layers[20] = make_shortcut_layer(layers[17], 0, "relu");

    layers[21] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "linear");
    layers[22] = make_normalization_layer(0.1, 1, "relu");
    layers[23] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "linear");
    layers[24] = make_normalization_layer(0.1, 1, "linear");
    layers[25] = make_shortcut_layer(layers[21], 0, "relu");

    layers[26] = make_convolutional_layer(256, 3, 2, 1, 0, 0, "linear");
    layers[27] = make_normalization_layer(0.1, 1, "relu");
    layers[28] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "linear");
    layers[29] = make_normalization_layer(0.1, 1, "linear");
    layers[30] = make_shortcut_layer(layers[26], 1, "linear");
    layers[31] = make_convolutional_layer(256, 1, 2, 0, 0, 0, "linear");
    layers[32] = make_normalization_layer(0.1, 1, "linear");
    layers[33] = make_shortcut_layer(layers[30], 0, "relu");

    layers[34] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "linear");
    layers[35] = make_normalization_layer(0.1, 1, "relu");
    layers[36] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "linear");
    layers[37] = make_normalization_layer(0.1, 1, "linear");
    layers[38] = make_shortcut_layer(layers[34], 0, "relu");

    layers[39] = make_convolutional_layer(512, 3, 2, 1, 0, 0, "linear");
    layers[40] = make_normalization_layer(0.1, 1, "relu");
    layers[41] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[42] = make_normalization_layer(0.1, 1, "linear");
    layers[43] = make_shortcut_layer(layers[39], 1, "linear");
    layers[44] = make_convolutional_layer(512, 1, 2, 0, 0, 0, "linear");
    layers[45] = make_normalization_layer(0.1, 1, "linear");
    layers[46] = make_shortcut_layer(layers[43], 0, "relu");

    layers[47] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[48] = make_normalization_layer(0.1, 1, "relu");
    layers[49] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[50] = make_normalization_layer(0.1, 1, "linear");
    layers[51] = make_shortcut_layer(layers[47], 0, "relu");
    // SPP
    layers[52] = make_maxpool_layer(5, 1, 2);
    layers[53] = make_shortcut_layer(layers[52], 1, "linear");
    layers[54] = make_maxpool_layer(9, 1, 4);
    layers[55] = make_shortcut_layer(layers[52], 1, "linear");
    layers[56] = make_maxpool_layer(13, 1, 6);
    Layer **spp = malloc(4*sizeof(Layer*));
    layers[57] = make_inception_layer(spp, 4, 2);
    spp[0] = layers[52];
    spp[1] = layers[53];
    spp[2] = layers[55];
    spp[3] = layers[57];
    // yolo检测头
    layers[58] = make_convolutional_layer(512, 1, 1, 0, 0, 1, "linear");
    layers[59] = make_normalization_layer(0.1, 1, "leaky");
    layers[60] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "linear");
    layers[61] = make_normalization_layer(0.1, 1, "leaky");
    layers[62] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "linear");
    layers[63] = make_normalization_layer(0.1, 1, "leaky");
    layers[64] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "linear");
    layers[65] = make_normalization_layer(0.1, 1, "leaky");
    layers[66] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "linear");
    layers[67] = make_normalization_layer(0.1, 1, "leaky");
    layers[68] = make_convolutional_layer(25, 1, 1, 0, 0, 1, "linear");
    // 只做推理时删除
    layers[69] = make_yolo_layer(size, stride);

    for (int i = 0; i < 70; ++i) {
        append_layer2grpah(graph, layers[i]);
        Layer *l = layers[i];
        if (l->type == CONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, sqrt(5.0), "fan_in", "relu");
            init_constant_bias(l, 0);
        }
    }

    Session *sess = create_session(graph, 416, 416, 3, grids*grids*(num_bbox*5+2), num_classes, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 416, 416);
    set_train_params(sess, 100, 16, 16, 0.01);//0.00001
    SGDOptimizer_sess(sess, 0.9, 0, 5e-4, 0, 0);
    // lr_scheduler_step(sess, 20, 0.5);
    init_session(sess, "./data/VOC2012/object.txt", "./data/VOC2012/object_label.txt");
    train(sess);
}

void yolov1_detect(char *type, char *path)
{
    int stride = 32;
    int num_bbox = 1;
    int num_classes = 20;
    int size = 416;
    int grids = size / stride;
    Graph *graph = create_graph();
    Layer **layers = malloc(69*sizeof(Layer*));
    // backbone-resnet18
    layers[0] = make_convolutional_layer(64, 7, 2, 3, 0, 0, "linear");
    layers[1] = make_normalization_layer(0.1, 1, "relu");
    layers[2] = make_maxpool_layer(3, 2, 1);

    layers[3] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "linear");
    layers[4] = make_normalization_layer(0.1, 1, "relu");
    layers[5] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "linear");
    layers[6] = make_normalization_layer(0.1, 1, "linear");
    layers[7] = make_shortcut_layer(layers[3], 0, "relu");

    layers[8] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "linear");
    layers[9] = make_normalization_layer(0.1, 1, "relu");
    layers[10] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "linear");
    layers[11] = make_normalization_layer(0.1, 1, "linear");
    layers[12] = make_shortcut_layer(layers[8], 0, "relu");

    layers[13] = make_convolutional_layer(128, 3, 2, 1, 0, 0, "linear");
    layers[14] = make_normalization_layer(0.1, 1, "relu");
    layers[15] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "linear");
    layers[16] = make_normalization_layer(0.1, 1, "linear");
    layers[17] = make_shortcut_layer(layers[13], 1, "linear");
    layers[18] = make_convolutional_layer(128, 1, 2, 0, 0, 0, "linear");
    layers[19] = make_normalization_layer(0.1, 1, "linear");
    layers[20] = make_shortcut_layer(layers[17], 0, "relu");

    layers[21] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "linear");
    layers[22] = make_normalization_layer(0.1, 1, "relu");
    layers[23] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "linear");
    layers[24] = make_normalization_layer(0.1, 1, "linear");
    layers[25] = make_shortcut_layer(layers[21], 0, "relu");

    layers[26] = make_convolutional_layer(256, 3, 2, 1, 0, 0, "linear");
    layers[27] = make_normalization_layer(0.1, 1, "relu");
    layers[28] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "linear");
    layers[29] = make_normalization_layer(0.1, 1, "linear");
    layers[30] = make_shortcut_layer(layers[26], 1, "linear");
    layers[31] = make_convolutional_layer(256, 1, 2, 0, 0, 0, "linear");
    layers[32] = make_normalization_layer(0.1, 1, "linear");
    layers[33] = make_shortcut_layer(layers[30], 0, "relu");

    layers[34] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "linear");
    layers[35] = make_normalization_layer(0.1, 1, "relu");
    layers[36] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "linear");
    layers[37] = make_normalization_layer(0.1, 1, "linear");
    layers[38] = make_shortcut_layer(layers[34], 0, "relu");

    layers[39] = make_convolutional_layer(512, 3, 2, 1, 0, 0, "linear");
    layers[40] = make_normalization_layer(0.1, 1, "relu");
    layers[41] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[42] = make_normalization_layer(0.1, 1, "linear");
    layers[43] = make_shortcut_layer(layers[39], 1, "linear");
    layers[44] = make_convolutional_layer(512, 1, 2, 0, 0, 0, "linear");
    layers[45] = make_normalization_layer(0.1, 1, "linear");
    layers[46] = make_shortcut_layer(layers[43], 0, "relu");

    layers[47] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[48] = make_normalization_layer(0.1, 1, "relu");
    layers[49] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "linear");
    layers[50] = make_normalization_layer(0.1, 1, "linear");
    layers[51] = make_shortcut_layer(layers[47], 0, "relu");
    // SPP
    layers[52] = make_maxpool_layer(5, 1, 2);
    layers[53] = make_shortcut_layer(layers[52], 1, "linear");
    layers[54] = make_maxpool_layer(9, 1, 4);
    layers[55] = make_shortcut_layer(layers[52], 1, "linear");
    layers[56] = make_maxpool_layer(13, 1, 6);
    Layer **spp = malloc(4*sizeof(Layer*));
    layers[57] = make_inception_layer(spp, 4, 2);
    spp[0] = layers[52];
    spp[1] = layers[53];
    spp[2] = layers[55];
    spp[3] = layers[57];
    // yolo检测头
    layers[58] = make_convolutional_layer(512, 1, 1, 0, 0, 1, "linear");
    layers[59] = make_normalization_layer(0.1, 1, "leaky");
    layers[60] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "linear");
    layers[61] = make_normalization_layer(0.1, 1, "leaky");
    layers[62] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "linear");
    layers[63] = make_normalization_layer(0.1, 1, "leaky");
    layers[64] = make_convolutional_layer(256, 1, 1, 0, 0, 1, "linear");
    layers[65] = make_normalization_layer(0.1, 1, "leaky");
    layers[66] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "linear");
    layers[67] = make_normalization_layer(0.1, 1, "leaky");
    layers[68] = make_convolutional_layer(25, 1, 1, 0, 0, 1, "linear");

    for (int i = 0; i < 69; ++i) {
        append_layer2grpah(graph, layers[i]);
    }

    Session *sess = create_session(graph, 416, 416, 3, grids*grids*(num_bbox*5+2), num_classes, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 416, 416);
    set_detect_params(sess);
    init_session(sess, "./data/VOC2012/test.txt", "./data/VOC2012/object_label.txt");
    detect_object(sess);
}
