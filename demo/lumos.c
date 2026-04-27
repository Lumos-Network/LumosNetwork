#include "xor.h"
#include "resnet18.h"
#include "cifar.h"
#include "darknet.h"
#include "alexnet_flower.h"
#include "lenet5_fmnist.h"
#include "lenet5_mnist.h"

int main()
{
    // xor("cpu", NULL);
    // xor_detect("cpu", "./backup/LW_f");
    // xor_detect("cpu", "./demo/xor.lw");
    // resnet18("gpu", "./backup/LW_py");
    // cifar("gpu", NULL);
    // darknet("gpu", "./backup/darknet");
    // alexnet_flower("gpu", NULL);
    // lenet5_fmnist("gpu", NULL);
    // lenet5_fmnist_detect("gpu", "./backup/LW_f");
    lenet5_mnist("gpu", NULL);
    lenet5_mnist_detect("gpu", "./backup/LW_f");
}
