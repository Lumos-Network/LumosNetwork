#ifndef RESNET50_H
#define RESNET50_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void resnet50(char *type, char *path);
void resnet50_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif