#ifndef YOLOV1_H
#define YOLOV1_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void yolov1(char *type, char *path);
void yolov1_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif