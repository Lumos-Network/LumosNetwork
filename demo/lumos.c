#include "xor.h"
#include "resnet18.h"
#include "cifar.h"
#include "darknet.h"
#include "alexnet_flower.h"
#include "lenet5_fmnist.h"
#include "lenet5_mnist.h"
#include "cpu.h"
#include "googlenet.h"
#include "fcn8.h"

int main()
{
    // Graph *g = create_graph();
    // Layer *l0 = make_convolutional_layer(3, 3, 1, 1, 1, "relu");
    // Layer *l1 = make_deconvolutional_layer(16, 3, 2, 1, 1, "relu");
    // // Layer *l2 = make_maxpool_layer(2, 2, 0);
    // // Layer *l3 = make_maxpool_layer(2, 2, 0);
    // // Layer *l4 = make_maxpool_layer(2, 2, 0);
    // // Layer *l5 = make_maxpool_layer(2, 2, 0);
    // Layer *l6 = make_connect_layer(5, 1, "linear");
    // Layer *l7 = make_crossentropy_layer(NULL, -1);
    // append_layer2grpah(g, l0);
    // append_layer2grpah(g, l1);
    // // append_layer2grpah(g, l2);
    // // append_layer2grpah(g, l3);
    // // append_layer2grpah(g, l4);
    // // append_layer2grpah(g, l5);
    // append_layer2grpah(g, l6);
    // append_layer2grpah(g, l7);
    // Session *sess = create_session(g, 224, 224, 3, 1, 5, "gpu", "./backup/LW_py");
    // float *mean = calloc(3, sizeof(float));
    // float *std = calloc(3, sizeof(float));
    // mean[0] = 0.485;
    // mean[1] = 0.456;
    // mean[2] = 0.406;
    // std[0] = 0.229;
    // std[1] = 0.224;
    // std[2] = 0.225;
    // transform_normalize_sess(sess, mean, std);
    // transform_resize_sess(sess, 224, 224);
    // set_train_params(sess, 4, 4, 4, 0.001);
    // SGDOptimizer_sess(sess, 0.9, 0, 0, 0, 0);
    // init_session(sess, "./data/flower/train_c.txt", "./data/flower/train_c_label.txt");
    // train(sess);
    // fcn8("gpu", NULL);
    lenet5_mnist("gpu", NULL);
}
