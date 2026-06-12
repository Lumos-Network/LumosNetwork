#include "resnet50.h"

void resnet50(char *type, char *path)
{
    int num_class = 5;
    Graph *graph = create_graph();
    Layer **layers = malloc(77*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 7, 2, 3, 0, 0, "relu");
    layers[1] = make_maxpool_layer(3, 2, 1);

    layers[2] = make_convolutional_layer(64, 1, 1, 0, 0, 0, "relu");
    layers[3] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "relu");
    layers[4] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "linear");
    layers[5] = make_shortcut_layer(layers[2], 1, "linear");
    layers[6] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "linear");
    layers[7] = make_shortcut_layer(layers[5], 0, "relu");

    layers[8] = make_convolutional_layer(64, 1, 1, 0, 0, 0, "relu");
    layers[9] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "relu");
    layers[10] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "linear");
    layers[11] = make_shortcut_layer(layers[8], 0, "relu");

    layers[12] = make_convolutional_layer(64, 1, 1, 0, 0, 0, "relu");
    layers[13] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "relu");
    layers[14] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[15] = make_shortcut_layer(layers[12], 0, "relu");

    layers[16] = make_convolutional_layer(128, 1, 1, 0, 0, 0, "relu");
    layers[17] = make_convolutional_layer(128, 3, 2, 1, 0, 0, "relu");
    layers[18] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "linear");
    layers[19] = make_shortcut_layer(layers[16], 1, "linear");
    layers[20] = make_convolutional_layer(512, 1, 2, 0, 0, 0, "linear");
    layers[21] = make_shortcut_layer(layers[19], 0, "relu");

    layers[22] = make_convolutional_layer(128, 1, 1, 0, 0, 0, "relu");
    layers[23] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "relu");
    layers[24] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "linear");
    layers[25] = make_shortcut_layer(layers[22], 0, "relu");

    layers[26] = make_convolutional_layer(128, 1, 1, 0, 0, 0, "relu");
    layers[27] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "relu");
    layers[28] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "linear");
    layers[29] = make_shortcut_layer(layers[26], 0, "relu");

    layers[30] = make_convolutional_layer(128, 1, 1, 0, 0, 0, "relu");
    layers[31] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "relu");
    layers[32] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "linear");
    layers[33] = make_shortcut_layer(layers[30], 0, "relu");

    layers[34] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[35] = make_convolutional_layer(256, 3, 2, 1, 0, 0, "relu");
    layers[36] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[37] = make_shortcut_layer(layers[34], 1, "linear");
    layers[38] = make_convolutional_layer(1024, 1, 2, 0, 0, 0, "linear");
    layers[39] = make_shortcut_layer(layers[37], 0, "relu");

    layers[40] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[41] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[42] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[43] = make_shortcut_layer(layers[40], 0, "relu");

    layers[44] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[45] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[46] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[47] = make_shortcut_layer(layers[44], 0, "relu");

    layers[48] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[49] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[50] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[51] = make_shortcut_layer(layers[48], 0, "relu");

    layers[52] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[53] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[54] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[55] = make_shortcut_layer(layers[52], 0, "relu");

    layers[56] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[57] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[58] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[59] = make_shortcut_layer(layers[56], 0, "relu");

    layers[60] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "relu");
    layers[61] = make_convolutional_layer(512, 3, 2, 1, 0, 0, "relu");
    layers[62] = make_convolutional_layer(2048, 1, 1, 0, 0, 0, "linear");
    layers[63] = make_shortcut_layer(layers[60], 1, "linear");
    layers[64] = make_convolutional_layer(2048, 1, 2, 0, 0, 0, "linear");
    layers[65] = make_shortcut_layer(layers[63], 0, "relu");

    layers[66] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "relu");
    layers[67] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "relu");
    layers[68] = make_convolutional_layer(2048, 1, 1, 0, 0, 0, "linear");
    layers[69] = make_shortcut_layer(layers[66], 0, "relu");

    layers[70] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "relu");
    layers[71] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "relu");
    layers[72] = make_convolutional_layer(2048, 1, 1, 0, 0, 0, "linear");
    layers[73] = make_shortcut_layer(layers[70], 0, "relu");

    layers[74] = make_avgpool_layer(7, 7, 0);
    layers[75] = make_connect_layer(num_class, 1, "linear");
    layers[76] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 77; ++i){
        append_layer2grpah(graph, layers[i]);
        Layer *l = layers[i];
        if (l->type == CONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, 0, "fan_in", "relu");
            init_constant_bias(l, 0);
        }
    }

    Session *sess = create_session(graph, 224, 224, 3, 1, num_class, type, path);
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
    set_train_params(sess, 50, 32, 32, 0.001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/flower/train.txt", "./data/flower/train_label.txt");
    train(sess);
}

void resnet50_detect(char*type, char *path)
{
    int num_class = 5;
    Graph *graph = create_graph();
    Layer **layers = malloc(77*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 7, 2, 3, 0, 0, "relu");
    layers[1] = make_maxpool_layer(3, 2, 1);

    layers[2] = make_convolutional_layer(64, 1, 1, 0, 0, 0, "relu");
    layers[3] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "relu");
    layers[4] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "linear");
    layers[5] = make_shortcut_layer(layers[2], 1, "linear");
    layers[6] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "linear");
    layers[7] = make_shortcut_layer(layers[5], 0, "relu");

    layers[8] = make_convolutional_layer(64, 1, 1, 0, 0, 0, "relu");
    layers[9] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "relu");
    layers[10] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "linear");
    layers[11] = make_shortcut_layer(layers[8], 0, "relu");

    layers[12] = make_convolutional_layer(64, 1, 1, 0, 0, 0, "relu");
    layers[13] = make_convolutional_layer(64, 3, 1, 1, 0, 0, "relu");
    layers[14] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[15] = make_shortcut_layer(layers[12], 0, "relu");

    layers[16] = make_convolutional_layer(128, 1, 1, 0, 0, 0, "relu");
    layers[17] = make_convolutional_layer(128, 3, 2, 1, 0, 0, "relu");
    layers[18] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "linear");
    layers[19] = make_shortcut_layer(layers[16], 1, "linear");
    layers[20] = make_convolutional_layer(512, 1, 2, 0, 0, 0, "linear");
    layers[21] = make_shortcut_layer(layers[19], 0, "relu");

    layers[22] = make_convolutional_layer(128, 1, 1, 0, 0, 0, "relu");
    layers[23] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "relu");
    layers[24] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "linear");
    layers[25] = make_shortcut_layer(layers[22], 0, "relu");

    layers[26] = make_convolutional_layer(128, 1, 1, 0, 0, 0, "relu");
    layers[27] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "relu");
    layers[28] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "linear");
    layers[29] = make_shortcut_layer(layers[26], 0, "relu");

    layers[30] = make_convolutional_layer(128, 1, 1, 0, 0, 0, "relu");
    layers[31] = make_convolutional_layer(128, 3, 1, 1, 0, 0, "relu");
    layers[32] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "linear");
    layers[33] = make_shortcut_layer(layers[30], 0, "relu");

    layers[34] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[35] = make_convolutional_layer(256, 3, 2, 1, 0, 0, "relu");
    layers[36] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[37] = make_shortcut_layer(layers[34], 1, "linear");
    layers[38] = make_convolutional_layer(1024, 1, 2, 0, 0, 0, "linear");
    layers[39] = make_shortcut_layer(layers[37], 0, "relu");

    layers[40] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[41] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[42] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[43] = make_shortcut_layer(layers[40], 0, "relu");

    layers[44] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[45] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[46] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[47] = make_shortcut_layer(layers[44], 0, "relu");

    layers[48] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[49] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[50] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[51] = make_shortcut_layer(layers[48], 0, "relu");

    layers[52] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[53] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[54] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[55] = make_shortcut_layer(layers[52], 0, "relu");

    layers[56] = make_convolutional_layer(256, 1, 1, 0, 0, 0, "relu");
    layers[57] = make_convolutional_layer(256, 3, 1, 1, 0, 0, "relu");
    layers[58] = make_convolutional_layer(1024, 1, 1, 0, 0, 0, "linear");
    layers[59] = make_shortcut_layer(layers[56], 0, "relu");

    layers[60] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "relu");
    layers[61] = make_convolutional_layer(512, 3, 2, 1, 0, 0, "relu");
    layers[62] = make_convolutional_layer(2048, 1, 1, 0, 0, 0, "linear");
    layers[63] = make_shortcut_layer(layers[60], 1, "linear");
    layers[64] = make_convolutional_layer(2048, 1, 2, 0, 0, 0, "linear");
    layers[65] = make_shortcut_layer(layers[63], 0, "relu");

    layers[66] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "relu");
    layers[67] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "relu");
    layers[68] = make_convolutional_layer(2048, 1, 1, 0, 0, 0, "linear");
    layers[69] = make_shortcut_layer(layers[66], 0, "relu");

    layers[70] = make_convolutional_layer(512, 1, 1, 0, 0, 0, "relu");
    layers[71] = make_convolutional_layer(512, 3, 1, 1, 0, 0, "relu");
    layers[72] = make_convolutional_layer(2048, 1, 1, 0, 0, 0, "linear");
    layers[73] = make_shortcut_layer(layers[70], 0, "relu");

    layers[74] = make_avgpool_layer(7, 7, 0);
    layers[75] = make_connect_layer(num_class, 1, "linear");
    layers[76] = make_crossentropy_layer(NULL, -1);

    for (int i = 0; i < 77; ++i){
        append_layer2grpah(graph, layers[i]);
    }

    Session *sess = create_session(graph, 224, 224, 3, 1, 5, type, path);
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
