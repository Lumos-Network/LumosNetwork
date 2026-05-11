#include "xor.h"
#include "resnet18.h"
#include "cifar.h"
#include "darknet.h"
#include "alexnet_flower.h"
#include "lenet5_fmnist.h"
#include "lenet5_mnist.h"
#include "cpu.h"
#include "googlenet.h"

int main()
{
    // googlenet("gpu", NULL);
    googlenet_detect("gpu", "./backup/LW_f");
}
