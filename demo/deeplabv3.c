#include "deeplabv3.h"

// 使用VGG16作为骨干网络
void deeplabv3(char *type, char *path)
{
    int num_class = 21;
    Graph *graph = create_graph();
    Layer **layers = malloc(36*sizeof(Layer*));
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
    layers[13] = make_maxpool_layer(2, 1, 1);
    // pool4
    layers[14] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[15] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[17] = make_maxpool_layer(2, 1, 1);
    //ASPP模块
    //x1
    layers[18] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    //x2
    layers[19] = make_shortcut_layer(layers[18], 1, "linear");
    layers[20] = make_convolutional_layer(256, 3, 1, 6, 6, 0, "relu");
    //x3
    layers[21] = make_shortcut_layer(layers[18], 1, "linear");
    layers[22] = make_convolutional_layer(256, 3, 1, 12, 12, 0, "relu");
    //x4
    layers[23] = make_shortcut_layer(layers[18], 1, "linear");
    layers[24] = make_convolutional_layer(256, 3, 1, 18, 18, 0, "relu");
    //x5
    layers[25] = make_shortcut_layer(layers[18], 1, "linear");
    layers[26] = make_avgpool_layer(36, 36, 0);
    layers[27] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[28] = make_interpolate_layer(36, 36);
    Layer **aspp = malloc(5*sizeof(Layer*));
    layers[29] = make_inception_layer(aspp, 5, 2);
    aspp[0] = layers[19];
    aspp[1] = layers[21];
    aspp[2] = layers[23];
    aspp[3] = layers[25];
    aspp[4] = layers[29];
    // project
    layers[30] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[31] = make_dropout_layer(0.5);

    layers[32] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[33] = make_convolutional_layer(num_class, 1, 1, 0, 0, 0, "linear");
    layers[34] = make_interpolate_layer(320, 320);
    layers[35] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 36; ++i){
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

void deeplabv3_detect(char*type, char *path)
{
    int num_class = 21;
    Graph *graph = create_graph();
    Layer **layers = malloc(36*sizeof(Layer*));
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
    layers[13] = make_maxpool_layer(2, 1, 1);
    // pool4
    layers[14] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[15] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[17] = make_maxpool_layer(2, 1, 1);
    //ASPP模块
    //x1
    layers[18] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    //x2
    layers[19] = make_shortcut_layer(layers[18], 1, "linear");
    layers[20] = make_convolutional_layer(256, 3, 1, 6, 6, 0, "relu");
    //x3
    layers[21] = make_shortcut_layer(layers[18], 1, "linear");
    layers[22] = make_convolutional_layer(256, 3, 1, 12, 12, 0, "relu");
    //x4
    layers[23] = make_shortcut_layer(layers[18], 1, "linear");
    layers[24] = make_convolutional_layer(256, 3, 1, 18, 18, 0, "relu");
    //x5
    layers[25] = make_shortcut_layer(layers[18], 1, "linear");
    layers[26] = make_avgpool_layer(36, 36, 0);
    layers[27] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[28] = make_interpolate_layer(36, 36);
    Layer **aspp = malloc(5*sizeof(Layer*));
    layers[29] = make_inception_layer(aspp, 5, 2);
    aspp[0] = layers[19];
    aspp[1] = layers[21];
    aspp[2] = layers[23];
    aspp[3] = layers[25];
    aspp[4] = layers[29];
    // project
    layers[30] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[31] = make_dropout_layer(0.5);

    layers[32] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[33] = make_convolutional_layer(num_class, 1, 1, 0, 0, 0, "linear");
    layers[34] = make_interpolate_layer(320, 320);
    layers[35] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 36; ++i){
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
