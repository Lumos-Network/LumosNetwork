#include "xor.h"
#include "resnet18.h"
#include "cifar.h"
#include "darknet.h"
#include "alexnet_flower.h"

int main()
{
    // xor("cpu", NULL);
    // xor_detect("cpu", "./backup/LW_f");
    // xor_detect("cpu", "./demo/xor.lw");
    // resnet18("gpu", "./backup/LW_py");
    // cifar("gpu", NULL);
    // darknet("gpu", NULL);
    alexnet_flower("gpu", NULL);
}
