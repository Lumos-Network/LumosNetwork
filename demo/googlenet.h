#ifndef GOOGLENET_H
#define GOOGLENET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void googlenet(char *type, char *path);
void googlenet_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
