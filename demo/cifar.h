#ifndef CIFAR_H
#define CIFAR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void cifar(char *type, char *path);
void cifar_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
