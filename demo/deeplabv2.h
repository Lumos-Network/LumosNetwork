#ifndef DEEPLABV2_H
#define DEEPLABV2_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void deeplabv2(char *type, char *path);
void deeplabv2_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif