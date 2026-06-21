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
#include "yolov1.h"
#include "darknet24.h"

#include "text_f.h"
#include "binary_f.h"
#include <math.h>
#include <string.h>

int main()
{
    yolov1("gpu", NULL);
    // darknet24("gpu", NULL);
    // lenet5_fmnist("gpu", NULL);
    // deeplabv1("gpu", NULL);
}
