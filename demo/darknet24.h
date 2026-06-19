#ifndef DARKNET24_H
#define DARKNET24_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void darknet24(char *type, char *path);
void darknet24_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif