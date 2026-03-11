#include "lenet5_mnist.h"
#include "xor.h"
#include "lenet5_cifar.h"
#include "lenet5_fmnist.h"
#include "alexnet.h"
#include "alexnet_flower.h"
#include "alexnet_xray.h"
#include "image.h"

#include "random.h"

int main()
{
    // lenet5_mnist("cpu", NULL);
    // lenet5_mnist_detect("gpu", "./backup/LW_f");

    // xor("gpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");

    // lenet5_cifar("gpu", NULL);
    // lenet5_cifar_detect("gpu", "./backup/LW_f");

    // lenet5_fmnist("gpu", NULL);
    // lenet5_fmnist_detect("gpu", "./backup/LW_f");

    // alexnet_flower("gpu", NULL);
    alexnet_flower_detect("gpu", "./backup/LW_f");

    // alexnet("gpu", NULL);
    // alexnet_detect("gpu", "./backup/LW_f");

    // alexnet_xray("gpu", "./backup/LW_py");
    // alexnet_xray_detect("gpu", "./backup/LW_f");

    return 0;
}