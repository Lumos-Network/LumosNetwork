#include "googlenet.h"

void googlenet(char *type, char *path)
{
    Graph *graph = create_graph();
    Layer **ls = malloc(112*sizeof(Layer*));
    ls[0] = make_convolutional_layer(64, 7, 2, 3, 0, "linear");
    ls[1] = make_normalization_layer(0.1, 1, "relu");
    ls[2] = make_maxpool_layer(3, 2, 1);
    ls[3] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");
    ls[4] = make_convolutional_layer(192, 3, 1, 1, 0, "linear");
    ls[5] = make_normalization_layer(0.1, 1, "relu");
    ls[6] = make_maxpool_layer(3, 2, 1);

    ls[7] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    ls[8] = make_shortcut_layer(ls[7], 1, "linear");
    ls[9] = make_convolutional_layer(96, 1, 1, 0, 1, "relu");
    ls[10] = make_convolutional_layer(128, 3, 1, 1, 1, "relu");

    ls[11] = make_shortcut_layer(ls[7], 1, "linear");
    ls[12] = make_convolutional_layer(16, 1, 1, 0, 1, "relu");
    ls[13] = make_convolutional_layer(32, 5, 1, 2, 1, "relu");

    ls[14] = make_shortcut_layer(ls[7], 1, "linear");
    ls[15] = make_maxpool_layer(3, 1, 1);
    ls[16] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");

    Layer **inception3a = malloc(4*sizeof(Layer*));
    ls[17] = make_inception_layer(inception3a, 4, 2);
    inception3a[0] = ls[8];
    inception3a[1] = ls[11];
    inception3a[2] = ls[14];
    inception3a[3] = ls[17];

    ls[18] = make_convolutional_layer(128, 1, 1, 0, 1, "relu"); // 16

    ls[19] = make_shortcut_layer(ls[18], 1, "linear");
    ls[20] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");
    ls[21] = make_convolutional_layer(192, 3, 1, 1, 1, "relu");

    ls[22] = make_shortcut_layer(ls[18], 1, "linear");
    ls[23] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
    ls[24] = make_convolutional_layer(96, 5, 1, 2, 1, "relu");

    ls[25] = make_shortcut_layer(ls[18], 1, "linear");
    ls[26] = make_maxpool_layer(3, 1, 1);
    ls[27] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception3b = malloc(4*sizeof(Layer*));
    ls[28] = make_inception_layer(inception3b, 4, 2);
    inception3b[0] = ls[19];
    inception3b[1] = ls[22];
    inception3b[2] = ls[25];
    inception3b[3] = ls[28];

    ls[29] = make_maxpool_layer(3, 2, 1); // 27

    ls[30] = make_convolutional_layer(192, 1, 1, 0, 1, "relu");

    ls[31] = make_shortcut_layer(ls[30], 1, "linear");
    ls[32] = make_convolutional_layer(96, 1, 1, 0, 1, "relu");
    ls[33] = make_convolutional_layer(208, 3, 1, 1, 1, "relu");

    ls[34] = make_shortcut_layer(ls[30], 1, "linear");
    ls[35] = make_convolutional_layer(16, 1, 1, 0, 1, "relu");
    ls[36] = make_convolutional_layer(48, 5, 1, 2, 1, "relu");

    ls[37] = make_shortcut_layer(ls[30], 1, "linear");
    ls[38] = make_maxpool_layer(3, 1, 1);
    ls[39] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception4a = malloc(4*sizeof(Layer*));
    ls[40] = make_inception_layer(inception4a, 4, 2);
    inception4a[0] = ls[31];
    inception4a[1] = ls[34];
    inception4a[2] = ls[37];
    inception4a[3] = ls[40];

    // 辅助分类器位置
    ls[41] = make_convolutional_layer(160, 1, 1, 0, 1, "relu"); // 39

    ls[42] = make_shortcut_layer(ls[41], 1, "linear");
    ls[43] = make_convolutional_layer(112, 1, 1, 0, 1, "relu");
    ls[44] = make_convolutional_layer(224, 3, 1, 1, 1, "relu");

    ls[45] = make_shortcut_layer(ls[41], 1, "linear");
    ls[46] = make_convolutional_layer(24, 1, 1, 0, 1, "relu");
    ls[47] = make_convolutional_layer(64, 5, 1, 2, 1, "relu");

    ls[48] = make_shortcut_layer(ls[41], 1, "linear");
    ls[49] = make_maxpool_layer(3, 1, 1);
    ls[50] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception4b = malloc(4*sizeof(Layer*));
    ls[51] = make_inception_layer(inception4b, 4, 2);
    inception4b[0] = ls[42];
    inception4b[1] = ls[45];
    inception4b[2] = ls[48];
    inception4b[3] = ls[51];

    ls[52] = make_convolutional_layer(128, 1, 1, 0, 1, "relu"); // 50

    ls[53] = make_shortcut_layer(ls[52], 1, "linear");
    ls[54] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");
    ls[55] = make_convolutional_layer(256, 3, 1, 1, 1, "relu");

    ls[56] = make_shortcut_layer(ls[52], 1, "linear");
    ls[57] = make_convolutional_layer(24, 1, 1, 0, 1, "relu");
    ls[58] = make_convolutional_layer(64, 5, 1, 2, 1, "relu");

    ls[59] = make_shortcut_layer(ls[52], 1, "linear");
    ls[60] = make_maxpool_layer(3, 1, 1);
    ls[61] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception4c = malloc(4*sizeof(Layer*));
    ls[62] = make_inception_layer(inception4c, 4, 2);
    inception4c[0] = ls[53];
    inception4c[1] = ls[56];
    inception4c[2] = ls[59];
    inception4c[3] = ls[62];

    ls[63] = make_convolutional_layer(112, 1, 1, 0, 1, "relu"); // 61

    ls[64] = make_shortcut_layer(ls[63], 1, "linear");
    ls[65] = make_convolutional_layer(144, 1, 1, 0, 1, "relu");
    ls[66] = make_convolutional_layer(288, 3, 1, 1, 1, "relu");

    ls[67] = make_shortcut_layer(ls[63], 1, "linear");
    ls[68] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
    ls[69] = make_convolutional_layer(64, 5, 1, 2, 1, "relu");

    ls[70] = make_shortcut_layer(ls[63], 1, "linear");
    ls[71] = make_maxpool_layer(3, 1, 1);
    ls[72] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception4d = malloc(4*sizeof(Layer*));
    ls[73] = make_inception_layer(inception4d, 4, 2);
    inception4d[0] = ls[64];
    inception4d[1] = ls[67];
    inception4d[2] = ls[70];
    inception4d[3] = ls[73];

    // 辅助分类器位置
    ls[74] = make_convolutional_layer(256, 1, 1, 0, 1, "relu"); // 72

    ls[75] = make_shortcut_layer(ls[74], 1, "linear");
    ls[76] = make_convolutional_layer(160, 1, 1, 0, 1, "relu");
    ls[77] = make_convolutional_layer(320, 3, 1, 1, 1, "relu");

    ls[78] = make_shortcut_layer(ls[74], 1, "linear");
    ls[79] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
    ls[80] = make_convolutional_layer(128, 5, 1, 2, 1, "relu");

    ls[81] = make_shortcut_layer(ls[74], 1, "linear");
    ls[82] = make_maxpool_layer(3, 1, 1);
    ls[83] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");

    Layer **inception4e = malloc(4*sizeof(Layer*));
    ls[84] = make_inception_layer(inception4e, 4, 2);
    inception4e[0] = ls[75];
    inception4e[1] = ls[78];
    inception4e[2] = ls[81];
    inception4e[3] = ls[84];

    ls[85] = make_maxpool_layer(2, 2, 0); // 83 25

    ls[86] = make_convolutional_layer(256, 1, 1, 0, 1, "relu");

    ls[87] = make_shortcut_layer(ls[86], 1, "linear");
    ls[88] = make_convolutional_layer(160, 1, 1, 0, 1, "relu");
    ls[89] = make_convolutional_layer(320, 3, 1, 1, 1, "relu");

    ls[90] = make_shortcut_layer(ls[86], 1, "linear");
    ls[91] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
    ls[92] = make_convolutional_layer(128, 5, 1, 2, 1, "relu");

    ls[93] = make_shortcut_layer(ls[86], 1, "linear");
    ls[94] = make_maxpool_layer(3, 1, 1);
    ls[95] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");

    Layer **inception5a = malloc(4*sizeof(Layer*));
    ls[96] = make_inception_layer(inception5a, 4, 2);
    inception5a[0] = ls[87];
    inception5a[1] = ls[90];
    inception5a[2] = ls[93];
    inception5a[3] = ls[96];

    ls[97] = make_convolutional_layer(384, 1, 1, 0, 1, "relu"); // 95

    ls[98] = make_shortcut_layer(ls[97], 1, "linear");
    ls[99] = make_convolutional_layer(192, 1, 1, 0, 1, "relu");
    ls[100] = make_convolutional_layer(384, 3, 1, 1, 1, "relu");

    ls[101] = make_shortcut_layer(ls[97], 1, "linear");
    ls[102] = make_convolutional_layer(48, 1, 1, 0, 1, "relu");
    ls[103] = make_convolutional_layer(128, 5, 1, 2, 1, "relu");

    ls[104] = make_shortcut_layer(ls[97], 1, "linear");
    ls[105] = make_maxpool_layer(3, 1, 1);
    ls[106] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");

    Layer **inception5b = malloc(4*sizeof(Layer*));
    ls[107] = make_inception_layer(inception5b, 4, 2);
    inception5b[0] = ls[98];
    inception5b[1] = ls[101];
    inception5b[2] = ls[104];
    inception5b[3] = ls[107];

    ls[108] = make_avgpool_layer(7, 7, 0); // 106
    ls[109] = make_dropout_layer(0.5);
    ls[110] = make_connect_layer(100, 1, "linear");
    ls[111] = make_crossentropy_layer(100);

    for (int i = 0; i < 112; ++i) {
        append_layer2grpah(graph, ls[i]);
        Layer *l = ls[i];
        if (l->type == CONVOLUTIONAL){
            init_kaiming_uniform_kernel(l, sqrt(5.0), "fan_in", "relu");
            init_constant_bias(l, 0);
        }
        if (l->type == CONNECT){
            init_kaiming_normal_kernel(l, sqrt(5.0), "fan_in", "relu");
            init_constant_bias(l, 0);
        }
    }

    Session *sess = create_session(graph, 96, 96, 3, 100, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 96, 96);
    set_train_params(sess, 150, 64, 64, 0.0001);
    SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    init_session(sess, "./data/cifar100/train.txt", "./data/cifar100/train_label.txt");
    train(sess);
}

void googlenet_detect(char*type, char *path)
{
    Graph *graph = create_graph();
    Layer **ls = malloc(112*sizeof(Layer*));
    ls[0] = make_convolutional_layer(64, 7, 2, 3, 0, "linear");
    ls[1] = make_normalization_layer(0.1, 1, "relu");
    ls[2] = make_maxpool_layer(3, 2, 1);
    ls[3] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");
    ls[4] = make_convolutional_layer(192, 3, 1, 1, 0, "linear");
    ls[5] = make_normalization_layer(0.1, 1, "relu");
    ls[6] = make_maxpool_layer(3, 2, 1);

    ls[7] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    ls[8] = make_shortcut_layer(ls[7], 1, "linear");
    ls[9] = make_convolutional_layer(96, 1, 1, 0, 1, "relu");
    ls[10] = make_convolutional_layer(128, 3, 1, 1, 1, "relu");

    ls[11] = make_shortcut_layer(ls[7], 1, "linear");
    ls[12] = make_convolutional_layer(16, 1, 1, 0, 1, "relu");
    ls[13] = make_convolutional_layer(32, 5, 1, 2, 1, "relu");

    ls[14] = make_shortcut_layer(ls[7], 1, "linear");
    ls[15] = make_maxpool_layer(3, 1, 1);
    ls[16] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");

    Layer **inception3a = malloc(4*sizeof(Layer*));
    ls[17] = make_inception_layer(inception3a, 4, 2);
    inception3a[0] = ls[8];
    inception3a[1] = ls[11];
    inception3a[2] = ls[14];
    inception3a[3] = ls[17];

    ls[18] = make_convolutional_layer(128, 1, 1, 0, 1, "relu"); // 16

    ls[19] = make_shortcut_layer(ls[18], 1, "linear");
    ls[20] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");
    ls[21] = make_convolutional_layer(192, 3, 1, 1, 1, "relu");

    ls[22] = make_shortcut_layer(ls[18], 1, "linear");
    ls[23] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
    ls[24] = make_convolutional_layer(96, 5, 1, 2, 1, "relu");

    ls[25] = make_shortcut_layer(ls[18], 1, "linear");
    ls[26] = make_maxpool_layer(3, 1, 1);
    ls[27] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception3b = malloc(4*sizeof(Layer*));
    ls[28] = make_inception_layer(inception3b, 4, 2);
    inception3b[0] = ls[19];
    inception3b[1] = ls[22];
    inception3b[2] = ls[25];
    inception3b[3] = ls[28];

    ls[29] = make_maxpool_layer(3, 2, 1); // 27

    ls[30] = make_convolutional_layer(192, 1, 1, 0, 1, "relu");

    ls[31] = make_shortcut_layer(ls[30], 1, "linear");
    ls[32] = make_convolutional_layer(96, 1, 1, 0, 1, "relu");
    ls[33] = make_convolutional_layer(208, 3, 1, 1, 1, "relu");

    ls[34] = make_shortcut_layer(ls[30], 1, "linear");
    ls[35] = make_convolutional_layer(16, 1, 1, 0, 1, "relu");
    ls[36] = make_convolutional_layer(48, 5, 1, 2, 1, "relu");

    ls[37] = make_shortcut_layer(ls[30], 1, "linear");
    ls[38] = make_maxpool_layer(3, 1, 1);
    ls[39] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception4a = malloc(4*sizeof(Layer*));
    ls[40] = make_inception_layer(inception4a, 4, 2);
    inception4a[0] = ls[31];
    inception4a[1] = ls[34];
    inception4a[2] = ls[37];
    inception4a[3] = ls[40];

    // 辅助分类器位置
    ls[41] = make_convolutional_layer(160, 1, 1, 0, 1, "relu"); // 39

    ls[42] = make_shortcut_layer(ls[41], 1, "linear");
    ls[43] = make_convolutional_layer(112, 1, 1, 0, 1, "relu");
    ls[44] = make_convolutional_layer(224, 3, 1, 1, 1, "relu");

    ls[45] = make_shortcut_layer(ls[41], 1, "linear");
    ls[46] = make_convolutional_layer(24, 1, 1, 0, 1, "relu");
    ls[47] = make_convolutional_layer(64, 5, 1, 2, 1, "relu");

    ls[48] = make_shortcut_layer(ls[41], 1, "linear");
    ls[49] = make_maxpool_layer(3, 1, 1);
    ls[50] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception4b = malloc(4*sizeof(Layer*));
    ls[51] = make_inception_layer(inception4b, 4, 2);
    inception4b[0] = ls[42];
    inception4b[1] = ls[45];
    inception4b[2] = ls[48];
    inception4b[3] = ls[51];

    ls[52] = make_convolutional_layer(128, 1, 1, 0, 1, "relu"); // 50

    ls[53] = make_shortcut_layer(ls[52], 1, "linear");
    ls[54] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");
    ls[55] = make_convolutional_layer(256, 3, 1, 1, 1, "relu");

    ls[56] = make_shortcut_layer(ls[52], 1, "linear");
    ls[57] = make_convolutional_layer(24, 1, 1, 0, 1, "relu");
    ls[58] = make_convolutional_layer(64, 5, 1, 2, 1, "relu");

    ls[59] = make_shortcut_layer(ls[52], 1, "linear");
    ls[60] = make_maxpool_layer(3, 1, 1);
    ls[61] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception4c = malloc(4*sizeof(Layer*));
    ls[62] = make_inception_layer(inception4c, 4, 2);
    inception4c[0] = ls[53];
    inception4c[1] = ls[56];
    inception4c[2] = ls[59];
    inception4c[3] = ls[62];

    ls[63] = make_convolutional_layer(112, 1, 1, 0, 1, "relu"); // 61

    ls[64] = make_shortcut_layer(ls[63], 1, "linear");
    ls[65] = make_convolutional_layer(144, 1, 1, 0, 1, "relu");
    ls[66] = make_convolutional_layer(288, 3, 1, 1, 1, "relu");

    ls[67] = make_shortcut_layer(ls[63], 1, "linear");
    ls[68] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
    ls[69] = make_convolutional_layer(64, 5, 1, 2, 1, "relu");

    ls[70] = make_shortcut_layer(ls[63], 1, "linear");
    ls[71] = make_maxpool_layer(3, 1, 1);
    ls[72] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");

    Layer **inception4d = malloc(4*sizeof(Layer*));
    ls[73] = make_inception_layer(inception4d, 4, 2);
    inception4d[0] = ls[64];
    inception4d[1] = ls[67];
    inception4d[2] = ls[70];
    inception4d[3] = ls[73];

    // 辅助分类器位置
    ls[74] = make_convolutional_layer(256, 1, 1, 0, 1, "relu"); // 72

    ls[75] = make_shortcut_layer(ls[74], 1, "linear");
    ls[76] = make_convolutional_layer(160, 1, 1, 0, 1, "relu");
    ls[77] = make_convolutional_layer(320, 3, 1, 1, 1, "relu");

    ls[78] = make_shortcut_layer(ls[74], 1, "linear");
    ls[79] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
    ls[80] = make_convolutional_layer(128, 5, 1, 2, 1, "relu");

    ls[81] = make_shortcut_layer(ls[74], 1, "linear");
    ls[82] = make_maxpool_layer(3, 1, 1);
    ls[83] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");

    Layer **inception4e = malloc(4*sizeof(Layer*));
    ls[84] = make_inception_layer(inception4e, 4, 2);
    inception4e[0] = ls[75];
    inception4e[1] = ls[78];
    inception4e[2] = ls[81];
    inception4e[3] = ls[84];

    ls[85] = make_maxpool_layer(2, 2, 0); // 83 25

    ls[86] = make_convolutional_layer(256, 1, 1, 0, 1, "relu");

    ls[87] = make_shortcut_layer(ls[86], 1, "linear");
    ls[88] = make_convolutional_layer(160, 1, 1, 0, 1, "relu");
    ls[89] = make_convolutional_layer(320, 3, 1, 1, 1, "relu");

    ls[90] = make_shortcut_layer(ls[86], 1, "linear");
    ls[91] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
    ls[92] = make_convolutional_layer(128, 5, 1, 2, 1, "relu");

    ls[93] = make_shortcut_layer(ls[86], 1, "linear");
    ls[94] = make_maxpool_layer(3, 1, 1);
    ls[95] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");

    Layer **inception5a = malloc(4*sizeof(Layer*));
    ls[96] = make_inception_layer(inception5a, 4, 2);
    inception5a[0] = ls[87];
    inception5a[1] = ls[90];
    inception5a[2] = ls[93];
    inception5a[3] = ls[96];

    ls[97] = make_convolutional_layer(384, 1, 1, 0, 1, "relu"); // 95

    ls[98] = make_shortcut_layer(ls[97], 1, "linear");
    ls[99] = make_convolutional_layer(192, 1, 1, 0, 1, "relu");
    ls[100] = make_convolutional_layer(384, 3, 1, 1, 1, "relu");

    ls[101] = make_shortcut_layer(ls[97], 1, "linear");
    ls[102] = make_convolutional_layer(48, 1, 1, 0, 1, "relu");
    ls[103] = make_convolutional_layer(128, 5, 1, 2, 1, "relu");

    ls[104] = make_shortcut_layer(ls[97], 1, "linear");
    ls[105] = make_maxpool_layer(3, 1, 1);
    ls[106] = make_convolutional_layer(128, 1, 1, 0, 1, "relu");

    Layer **inception5b = malloc(4*sizeof(Layer*));
    ls[107] = make_inception_layer(inception5b, 4, 2);
    inception5b[0] = ls[98];
    inception5b[1] = ls[101];
    inception5b[2] = ls[104];
    inception5b[3] = ls[107];

    ls[108] = make_avgpool_layer(7, 7, 0); // 106
    ls[109] = make_dropout_layer(0.5);
    ls[110] = make_connect_layer(100, 1, "linear");
    ls[111] = make_crossentropy_layer(100);

    for (int i = 0; i < 112; ++i) {
        append_layer2grpah(graph, ls[i]);
    }

    Session *sess = create_session(graph, 96, 96, 3, 100, type, path);
    float *mean = calloc(3, sizeof(float));
    float *std = calloc(3, sizeof(float));
    mean[0] = 0.485;
    mean[1] = 0.456;
    mean[2] = 0.406;
    std[0] = 0.229;
    std[1] = 0.224;
    std[2] = 0.225;
    transform_normalize_sess(sess, mean, std);
    transform_resize_sess(sess, 96, 96);
    set_detect_params(sess);
    init_session(sess, "./data/cifar100/train.txt", "./data/cifar100/train_label.txt");
    detect_classification(sess);
}
