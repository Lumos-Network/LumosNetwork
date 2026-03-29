#include "lenet5_mnist.h"
#include "xor.h"
#include "lenet5_fmnist.h"
#include "alexnet_flower.h"
#include "image.h"

#include "random.h"
#include "normalization_layer.h"
#include "vgg16_cifar10.h"
#include "resnet18.h"

int main()
{
    // lenet5_mnist("gpu", NULL);
    // lenet5_mnist_detect("gpu", "./backup/LW_f");

    // xor("gpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");

    // lenet5_fmnist("gpu", NULL);
    // lenet5_fmnist_detect("gpu", "./backup/LW_f");

    // alexnet_flower("cpu", "./backup/LW_py");
    // alexnet_flower_detect("gpu", "./backup/LW_f");

    // alexnet_xray("gpu", NULL);
    // alexnet_xray_detect("gpu", "./backup/LW_f");

    // xor("cpu", NULL);
    // xor_detect("cpu", "./backup/LW_f");

    // vgg16_cifar10("gpu", NULL);

    // resnet18("gpu", NULL);
    // resnet18_detect("gpu", "./backup/LW_f");

    return 0;
}