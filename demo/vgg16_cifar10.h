#ifndef VGG16_CIFAR10_H
#define VGG16_CIFAR10_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void vgg16_cifar10(char *type, char *path);
void vgg16_cifar10_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
