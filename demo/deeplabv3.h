#ifndef DEEPLABV3_H
#define DEEPLABV3_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void deeplabv3(char *type, char *path);
void deeplabv3_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif