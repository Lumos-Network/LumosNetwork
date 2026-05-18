#ifndef DARKNET_H
#define DARKNET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void darknet(char *type, char *path);
void darknet_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif