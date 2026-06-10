#ifndef DEEPLABV1_H
#define DEEPLABV1_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void deeplabv1(char *type, char *path);
void deeplabv1_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif