#include "lenet5_mnist.h"
#include "xor.h"

int main()
{
    lenet5_mnist("cpu", NULL);
    lenet5_mnist_detect("cpu", "./backup/LW_f");

    // xor("gpu", NULL);
    // xor_detect("gpu", "./backup/LW_f");
    return 0;
}