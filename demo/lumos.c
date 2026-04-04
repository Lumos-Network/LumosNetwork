#include "xor.h"
#include "resnet18.h"
#include "cifar.h"
#include "darknet.h"

int main()
{
    // xor_detect("cpu", "./demo/xor.lw");
    // resnet18("gpu", "./backup/LW_py");
    // cifar("gpu", NULL);
    darknet("gpu", NULL);
}
