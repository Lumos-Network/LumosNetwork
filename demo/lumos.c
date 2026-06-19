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

int main()
{
    // darknet24("gpu", NULL);
    void **labels = load_label_txt("./01.txt");
    int *lindex = (int*)labels[0];
    char *tmp = (char*)labels[1];
    float *truth = malloc(5*sizeof(float));
    for (int j = 0; j < 5; ++j){
        truth[j] = atof(tmp+lindex[j+1]);
        printf("%f ", truth[j]);
    }
    printf("\n");
    free(lindex);
    free(tmp);
    free(labels);
}
