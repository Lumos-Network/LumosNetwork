#ifndef RESNET18_H
#define RESNET18_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void resnet18(char *type, char *path);
void resnet18_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif