#include "lenet5_mnist.h"
#include "xor.h"
#include "lenet5_cifar.h"
#include "lenet5_fmnist.h"
#include "alexnet.h"
#include "alexnet_flower.h"

int main()
{
    lenet5_mnist("gpu", NULL);
    lenet5_mnist_detect("gpu", "./backup/LW_f");

    // xor("gpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");

    // lenet5_cifar("gpu", NULL);
    // lenet5_cifar_detect("gpu", "./backup/LW_f");

    // lenet5_fmnist("gpu", NULL);
    // lenet5_fmnist_detect("gpu", "./backup/LW_f");

    // alexnet_flower("gpu", "./backup/LW_1_100");
    // alexnet_flower_detect("gpu", "./backup/LW_f");

    // alexnet("gpu", NULL);
    // alexnet_detect("gpu", "./backup/LW_f");

    return 0;
}