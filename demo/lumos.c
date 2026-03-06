#include "lenet5_mnist.h"
#include "xor.h"
#include "lenet5_cifar.h"
#include "lenet5_fmnist.h"
#include "alexnet.h"
#include "alexnet_flower.h"
#include "alexnet_xray.h"
#include "image.h"

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

    alexnet_flower("cpu", "./backup/LW_py");
    // alexnet_flower_detect("gpu", "./backup/LW_f");

    // alexnet("gpu", NULL);
    // alexnet_detect("gpu", "./backup/LW_f");

    // alexnet_xray("gpu", NULL);

    // float *input = (float*)calloc(224*224*3, sizeof(float));
    // FILE *fp = fopen("./backup/data/0", "rb");
    // if (fp == NULL) printf("NO FILE!\n");
    // fread(input, sizeof(float), 224*224*3, fp);
    // fclose(fp);

    // int h[1], w[1], c[1];
    // float *im = load_image_data("./data/flower/train/dandelion_215.png", w, h, c);
    // for (int i = 0; i < 224*224*3; ++i){
    //     if (input[i] != im[i]){
    //         printf("error!\n");
    //         break;
    //     }
    // }
    return 0;
}