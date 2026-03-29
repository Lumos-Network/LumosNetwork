#include "googlenet.h"

void alexnet_flower(char *type, char *path)
{
    Graph *g = create_graph();
    Layer **NETLAYERS = malloc(1555);
    NETLAYERS[0] = make_convolutional_layer(64, 7, 2, 3, 1, "relu");
    NETLAYERS[1] = make_maxpool_layer(3, 2, 1);
    NETLAYERS[2] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");
    NETLAYERS[3] = make_convolutional_layer(192, 3, 1, 1, 1, "relu");
    NETLAYERS[4] = make_maxpool_layer(3, 2, 1);

    NETLAYERS[5] = make_convolutional_layer(64, 1, 1, 0, 1, "relu");
    NETLAYERS[6] = make_shortcut_layer(NETLAYERS[5], 1, "linear");
    NETLAYERS[7] = make_convolutional_layer(96, 1, 1, 0, 1, "relu");
    NETLAYERS[8] = make_convolutional_layer(128, 3, 1, 1, 1, "relu");
    NETLAYERS[9] = make_shortcut_layer(NETLAYERS[5], 1, "linear");
    NETLAYERS[10] = make_convolutional_layer(16, 1, 1, 0, 1, "relu");
    NETLAYERS[11] = make_convolutional_layer(32, 5, 1, 2, 1, "relu");
    NETLAYERS[12] = make_shortcut_layer(NETLAYERS[5], 1, "linear");
    NETLAYERS[13] = make_maxpool_layer(3, 1, 1);
    NETLAYERS[14] = make_convolutional_layer(32, 1, 1, 0, 1, "relu");
}