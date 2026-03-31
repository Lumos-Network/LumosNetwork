<<<<<<< HEAD
#include "lenet5_mnist.h"

int main()
{
    lenet5_mnist("cpu", NULL);
    lenet5_mnist_detect("cpu", "./LuWeights");
=======
#include "xor.h"

int main()
{
    xor_detect("cpu", "./demo/xor.lw");
>>>>>>> develop
}
