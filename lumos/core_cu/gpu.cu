#include "gpu.h"

dim3 cuda_gridsize(size_t n){
    unsigned int k = (n-1) / BLOCK + 1;
    unsigned int x = k;
    unsigned int y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}
