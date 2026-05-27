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
#include "image.h"
#include "cpu.h"
#include "cpu_gpu.h"

int main()
{
    // fcn8("gpu", "./backup/LW_py");
    // googlenet("gpu", NULL);
    xor("gpu", NULL);
    xor_detect("gpu", "./backup/LW_f");
}
