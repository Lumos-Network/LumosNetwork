#include "deeplabv3.h"

// 使用VGG16作为骨干网络
void deeplabv3(char *type, char *path)
{
    int num_class = 21;
    Graph *graph = create_graph();
    Layer **layers = malloc(32*sizeof(Layer*));
    layers[0] = make_convolutional_layer(64, 3, 1, 1, 0, 1, "relu");
    layers[1] = make_convolutional_layer(64, 3, 1, 1, 0, 1, "relu");
    layers[2] = make_maxpool_layer(2, 2, 0);

    layers[3] = make_convolutional_layer(128, 3, 1, 1, 0, 1, "relu");
    layers[4] = make_convolutional_layer(128, 3, 1, 1, 0, 1, "relu");
    layers[5] = make_maxpool_layer(2, 2, 0);

    layers[6] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[7] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[8] = make_convolutional_layer(256, 3, 1, 1, 0, 1, "relu");
    layers[9] = make_maxpool_layer(2, 2, 0);
    // pool3
    layers[10] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[11] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[12] = make_convolutional_layer(512, 3, 1, 1, 0, 1, "relu");
    layers[13] = make_maxpool_layer(2, 1, 1);
    // pool4
    layers[14] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[15] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[16] = make_convolutional_layer(512, 3, 1, 2, 2, 1, "relu");
    layers[17] = make_maxpool_layer(2, 1, 1);
    //ASPP模块
    
}

void deeplabv3_detect(char*type, char *path)
{

}
