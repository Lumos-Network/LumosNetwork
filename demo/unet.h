#ifndef UNET_H
#define UNET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void unet(char *type, char *path);
void unet_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif