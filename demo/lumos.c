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
#include "vgg16_cifar10.h"
#include "unet.h"
#include "deeplabv1.h"
#include "deeplabv2.h"
#include "deeplabv3.h"

int main()
{
    deeplabv3("gpu", "./backup/LW_deeplabv2");
    // deeplabv2_detect("gpu", "./backup/LW_f");
    // deeplabv3("gpu", NULL);
}
