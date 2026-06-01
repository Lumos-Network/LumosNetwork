#include "unet.h"

void unet(char *type, char *path)
{
    int num_class = 2;
    Graph *graph = create_graph();
    Layer **layers = malloc(50*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    layers[1] = make_normalization_layer(0.1, 1, "relu");
    layers[2] = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    layers[3] = make_normalization_layer(0.1, 1, "relu");
    // x1 256*256*64
    layers[4] = make_maxpool_layer(2, 2, 0);
    layers[5] = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    layers[6] = make_normalization_layer(0.1, 1, "relu");
    layers[7] = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    layers[8] = make_normalization_layer(0.1, 1, "relu");
    // x2 128*128*128
    layers[9] = make_maxpool_layer(2, 2, 0);
    layers[10] = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    layers[11] = make_normalization_layer(0.1, 1, "relu");
    layers[12] = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    layers[13] = make_normalization_layer(0.1, 1, "relu");
    // x3 64*64*256
    layers[14] = make_maxpool_layer(2, 2, 0);
    layers[15] = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    layers[16] = make_normalization_layer(0.1, 1, "relu");
    layers[17] = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    layers[18] = make_normalization_layer(0.1, 1, "relu");
    // x4 32*32*512
    layers[19] = make_maxpool_layer(2, 2, 0);
    layers[20] = make_convolutional_layer(1024, 3, 1, 1, 1, "linear");
    layers[21] = make_normalization_layer(0.1, 1, "relu");
    layers[22] = make_convolutional_layer(1024, 3, 1, 1, 1, "linear");
    layers[23] = make_normalization_layer(0.1, 1, "relu");
    // x5 16*16*1024
    layers[24] = make_deconvolutional_layer(512, 2, 2, 0, 0, "linear");
    // 32*32*512
    Layer **up1 = malloc(2*sizeof(Layer*));
    layers[25] = make_inception_layer(up1, 2, 2);
    up1[0] = layers[19];
    up1[1] = layers[25];
    // 32*32*1024
    layers[26] = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    layers[27] = make_normalization_layer(0.1, 1, "relu");
    layers[28] = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    layers[29] = make_normalization_layer(0.1, 1, "relu");
    // 32*32*512
    layers[30] = make_deconvolutional_layer(256, 2, 2, 0, 0, "linear");
    // 64*64*256
    Layer **up2 = malloc(2*sizeof(Layer*));
    layers[31] = make_inception_layer(up2, 2, 2);
    up2[0] = layers[14];
    up2[1] = layers[31];
    // 64*64*512
    layers[32] = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    layers[33] = make_normalization_layer(0.1, 1, "relu");
    layers[34] = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    layers[35] = make_normalization_layer(0.1, 1, "relu");
    // 64*64*256
    layers[36] = make_deconvolutional_layer(128, 2, 2, 0, 0, "linear");
    // 128*128*128
    Layer **up3 = malloc(2*sizeof(Layer*));
    layers[37] = make_inception_layer(up3, 2, 2);
    up3[0] = layers[9];
    up3[1] = layers[37];
    // 128*128*256
    layers[38] = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    layers[39] = make_normalization_layer(0.1, 1, "relu");
    layers[40] = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    layers[41] = make_normalization_layer(0.1, 1, "relu");
    // 128*128*128
    layers[42] = make_deconvolutional_layer(64, 2, 2, 0, 0, "linear");
    // 256*256*64
    Layer **up4 = malloc(2*sizeof(Layer*));
    layers[43] = make_inception_layer(up4, 2, 2);
    up4[0] = layers[4];
    up4[1] = layers[43];
    // 256*256*128
    layers[44] = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    layers[45] = make_normalization_layer(0.1, 1, "relu");
    layers[46] = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    layers[47] = make_normalization_layer(0.1, 1, "relu");
    // 256*256*64
    layers[48] = make_convolutional_layer(num_class, 1, 1, 0, 1, "linear");
    // 256*256*2
    layers[49] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 50; ++i){
        append_layer2grpah(graph, layers[i]);
        Layer *l = layers[i];
        if (l->type == CONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, 0, "fan_out", "relu");
            init_constant_bias(l, 0);
        }
        if (l->type == DECONVOLUTIONAL){
            init_bilinearinterp_kernel(l);
        }
    }
    Session *sess = create_session(graph, 256, 256, 1, 256*256, num_class, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 256, 256);
    set_train_params(sess, 50, 4, 4, 0.001);
    SGDOptimizer_sess(sess, 0.99, 0, 0, 0, 0);
    lr_scheduler_exponential(sess, 0.95);
    init_session(sess, "./data/umd/train.txt", "./data/umd/train_label.txt");
    train(sess);
}

void unet_detect(char *type, char *path)
{
    int num_class = 2;
    Graph *graph = create_graph();
    Layer **layers = malloc(50*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    layers[1] = make_normalization_layer(0.1, 1, "relu");
    layers[2] = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    layers[3] = make_normalization_layer(0.1, 1, "relu");
    // x1 256*256*64
    layers[4] = make_maxpool_layer(2, 2, 0);
    layers[5] = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    layers[6] = make_normalization_layer(0.1, 1, "relu");
    layers[7] = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    layers[8] = make_normalization_layer(0.1, 1, "relu");
    // x2 128*128*128
    layers[9] = make_maxpool_layer(2, 2, 0);
    layers[10] = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    layers[11] = make_normalization_layer(0.1, 1, "relu");
    layers[12] = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    layers[13] = make_normalization_layer(0.1, 1, "relu");
    // x3 64*64*256
    layers[14] = make_maxpool_layer(2, 2, 0);
    layers[15] = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    layers[16] = make_normalization_layer(0.1, 1, "relu");
    layers[17] = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    layers[18] = make_normalization_layer(0.1, 1, "relu");
    // x4 32*32*512
    layers[19] = make_maxpool_layer(2, 2, 0);
    layers[20] = make_convolutional_layer(1024, 3, 1, 1, 1, "linear");
    layers[21] = make_normalization_layer(0.1, 1, "relu");
    layers[22] = make_convolutional_layer(1024, 3, 1, 1, 1, "linear");
    layers[23] = make_normalization_layer(0.1, 1, "relu");
    // x5 16*16*1024
    layers[24] = make_deconvolutional_layer(512, 2, 2, 0, 0, "linear");
    // 32*32*512
    Layer **up1 = malloc(2*sizeof(Layer*));
    layers[25] = make_inception_layer(up1, 2, 2);
    up1[0] = layers[19];
    up1[1] = layers[25];
    // 32*32*1024
    layers[26] = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    layers[27] = make_normalization_layer(0.1, 1, "relu");
    layers[28] = make_convolutional_layer(512, 3, 1, 1, 1, "linear");
    layers[29] = make_normalization_layer(0.1, 1, "relu");
    // 32*32*512
    layers[30] = make_deconvolutional_layer(256, 2, 2, 0, 0, "linear");
    // 64*64*256
    Layer **up2 = malloc(2*sizeof(Layer*));
    layers[31] = make_inception_layer(up2, 2, 2);
    up2[0] = layers[14];
    up2[1] = layers[31];
    // 64*64*512
    layers[32] = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    layers[33] = make_normalization_layer(0.1, 1, "relu");
    layers[34] = make_convolutional_layer(256, 3, 1, 1, 1, "linear");
    layers[35] = make_normalization_layer(0.1, 1, "relu");
    // 64*64*256
    layers[36] = make_deconvolutional_layer(128, 2, 2, 0, 0, "linear");
    // 128*128*128
    Layer **up3 = malloc(2*sizeof(Layer*));
    layers[37] = make_inception_layer(up3, 2, 2);
    up3[0] = layers[9];
    up3[1] = layers[37];
    // 128*128*256
    layers[38] = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    layers[39] = make_normalization_layer(0.1, 1, "relu");
    layers[40] = make_convolutional_layer(128, 3, 1, 1, 1, "linear");
    layers[41] = make_normalization_layer(0.1, 1, "relu");
    // 128*128*128
    layers[42] = make_deconvolutional_layer(64, 2, 2, 0, 0, "linear");
    // 256*256*64
    Layer **up4 = malloc(2*sizeof(Layer*));
    layers[43] = make_inception_layer(up4, 2, 2);
    up4[0] = layers[4];
    up4[1] = layers[43];
    // 256*256*128
    layers[44] = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    layers[45] = make_normalization_layer(0.1, 1, "relu");
    layers[46] = make_convolutional_layer(64, 3, 1, 1, 1, "linear");
    layers[47] = make_normalization_layer(0.1, 1, "relu");
    // 256*256*64
    layers[48] = make_convolutional_layer(num_class, 1, 1, 0, 1, "linear");
    // 256*256*2
    layers[49] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 50; ++i){
        append_layer2grpah(graph, layers[i]);
    }

    Session *sess = create_session(graph, 256, 256, 1, 256*256, num_class, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 256, 256);
    set_detect_params(sess);
    init_session(sess, "./data/umd/train.txt", "./data/umd/train_label.txt");
    detect_segmentation(sess);
}
