#include "lenet5_mnist.h"
#include "xor.h"
#include "lenet5_fmnist.h"
#include "alexnet_flower.h"
#include "image.h"

#include "random.h"
#include "normalization_layer.h"

int main()
{
    // lenet5_mnist("gpu", NULL);
    // lenet5_mnist_detect("gpu", "./backup/LW_f");

    // xor("gpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");

    // lenet5_cifar("gpu", NULL);
    // lenet5_cifar_detect("gpu", "./backup/LW_f");

    // lenet5_fmnist("gpu", NULL);
    // lenet5_fmnist_detect("gpu", "./backup/LW_f");

    alexnet_flower("gpu", "./backup/LW_py");
    // alexnet_flower_detect("gpu", "./backup/LW_f");

    // alexnet_xray("gpu", NULL);
    // alexnet_xray_detect("gpu", "./backup/LW_f");

    // xor("gpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");

    // Layer *l = make_normalization_layer(64, 1, "linear");
    // l->initialize(l, 45, 35, 64, 20);
    // l->weightinit(*l, NULL);
    // l->status = 1;
    // float *input = calloc(20*45*35*64, sizeof(float));

    // FILE *fp = fopen("./backup/bn_in", "rb");
    // fread(input, sizeof(float), 20*45*35*64, fp);
    // fclose(fp);

    // l->input = input;
    // l->forward(*l, 20);

    // FILE *ffp = fopen("./backup/bn_out", "wb");
    // fwrite(l->output, sizeof(float), 20*45*35*64, ffp);

    return 0;
}