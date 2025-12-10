#include "lenet5_mnist.h"

int main()
{
    lenet5_mnist("gpu", NULL);
    lenet5_mnist_detect("gpu", "./LuWeights");
}
