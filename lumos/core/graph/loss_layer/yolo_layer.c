#include "yolo_layer.h"

#define MAX(x, y) ((x) > (y) ? (x) : (y))

float conf_msewithlogitsloss(float pred, float target)
{
    float input = 1 / (1 + exp(-pred));
    if (input < 1e-4) input = 1e-4;
    if (input > 1.0-(1e-4)) input = 1.0-(1e-4);
    float pos_id = (target==1.0)?1.0:0;
    float neg_id = (target==0.0)?1.0:0;

    float pos_loss = pos_id * pow(input-target, 2);
    float neg_loss = neg_id * pow(input, 2);

    float conf_loss = 5.0*pos_loss + 1.0*neg_loss;
    return conf_loss;
}

void conf_msewithlogitsloss_gradient(float pred, float target, float *space)
{
    float input = 1 / (1 + exp(-pred));
    if (input < 1e-4) input = 1e-4;
    if (input > 1.0-(1e-4)) input = 1.0-(1e-4);

    float pos_id = (target==1.0)?1.0:0;
    float neg_id = (target==0.0)?1.0:0;

    float pos_delta = pos_id * 2*(input-target);
    float neg_delta = neg_id * 2*input;

    float conf_delta = 5.0*pos_delta + 1.0*neg_delta;
    conf_delta *= (1-input)*input;
    space[0] = conf_delta;
}

float xy_bcewithlogitsloss(float px, float py, float tx, float ty, float box_scale_weight)
{
    float x_loss = 0;
    float y_loss = 0;
    x_loss = -px*tx + log(1+exp(px));
    y_loss = -py*ty + log(1+exp(py));
    float xy_loss = (x_loss + y_loss)*box_scale_weight;
    return xy_loss;
}

void xy_bcewithlogitsloss_gradient(float px, float py, float tx, float ty, float box_scale_weight, float *spacex, float *spacey)
{
    float x_delta = 0;
    float y_delta = 0;
    x_delta = 1 / (1 + exp(-px)) - tx;
    y_delta = 1 / (1 + exp(-py)) - ty;
    spacex[0] = x_delta*box_scale_weight;
    spacey[0] = y_delta*box_scale_weight;
}

float wh_mseloss(float pw, float ph, float tw, float th, float box_scale_weight)
{
    float w_loss = powf(pw-tw, 2);
    float h_loss = powf(ph-th, 2);
    float wh_loss = (w_loss + h_loss)*box_scale_weight;
    return wh_loss;
}

void wh_mseloss_gradient(float pw, float ph, float tw, float th, float box_scale_weight, float *spacew, float *spaceh)
{
    float w_delta = 2*(pw-tw);
    float h_delta = 2*(ph-th);
    spacew[0] = w_delta*box_scale_weight;
    spaceh[0] = h_delta*box_scale_weight;
}

float class_crossentropy(float *data, float truth, int w, int h, int c, int index)
{
    int target = (int)truth;
    float max_val = -INFINITY;
    float sum_exp = 0;
    for (int i = 0; i < c; ++i){
        max_val = MAX(max_val, data[i*w*h+index]);
    }
    for (int i = 0; i < c; ++i){
        sum_exp += expf(data[i*w*h+index]-max_val);
    }
    float loss = (-data[target*w*h+index]+max_val+log(sum_exp));
    return loss;
}

void class_crossentropy_gradient(float *data, float truth, int w, int h, int c, int index, float *space)
{
    int target = (int)truth;
    float max_val = -INFINITY;
    float sum_exp = 0;
    for (int i = 0; i < c; ++i){
        max_val = MAX(max_val, data[i*w*h+index]);
    }
    for (int i = 0; i < c; ++i){
        space[i*w*h+index] = expf(data[i*w*h+index]-max_val);
        sum_exp += space[i*w*h+index];
    }
    for (int i = 0; i < c; ++i){
        if (i == target) space[i*w*h+index] = (space[i*w*h+index]/sum_exp-1);
        else space[i*w*h+index] = (space[i*w*h+index]/sum_exp);
    }
}

void yolo_loss(int num, int grids, float *output)
{
    float conf_loss = 0;
    float clas_loss = 0;
    float bbox_loss = 0;
    float total_loss = 0;
    for (int i = 0; i < num; ++i){
        for (int j = 0; j < grids*grids; ++j){
            float *out = output + i*grids*grids*4 + j*4;
            conf_loss += out[0];
            clas_loss += out[1];
            bbox_loss += out[2];
        }
    }
    conf_loss /= num;
    clas_loss /= num;
    bbox_loss /= num;
    total_loss = (conf_loss + clas_loss + bbox_loss)/8;
    printf("conf:%f cls:%f bbox:%f total:%f\n", conf_loss, clas_loss, bbox_loss, total_loss);
}

Layer *make_yolo_layer(int size, int stride)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = YOLO;
    l->ksize = size;
    l->stride = stride;

    l->initialize = init_yolo_layer;
    l->forward = forward_yolo_layer;
    l->backward = backward_yolo_layer;
    l->initializegpu = init_yolo_layer_gpu;
    l->forwardgpu = forward_yolo_layer_gpu;
    l->backwardgpu = backward_yolo_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->sgdoptimizer = NULL;
    l->sgdoptimizergpu = NULL;

    l->refresh = NULL;
    l->refreshgpu = NULL;

    l->saveweights = NULL;
    l->saveweightsgpu = NULL;

    l->zerogradlayer = zerograd_yolo_layer;
    l->zerogradlayergpu = zerograd_yolo_layer_gpu;

    fprintf(stderr, "YOLO    Layer    :    [output=%4d]\n", 1);
    return l;
}

void init_yolo_layer(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    int grids = l->ksize / l->stride;
    l->output_h = 1;
    l->output_w = 1;
    l->output_c = grids*grids*4;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = 0;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "YOLO    Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

// 7*7*(5*2+20) [box1_x,box1_y,box1_w,box1_h,box1_conf,box2_x,box2_y,box2_w,box2_h,box2_conf,cls0~cls19]
void forward_yolo_layer(Layer l, int num)
{
    int grids = l.ksize / l.stride;
    fill_cpu(l.output, num*l.outputs, 0, 1);
    for (int index = 0; index < num*grids*grids; ++index)
    {
        int img_index = index / (grids*grids);
        int box_index = index % (grids*grids);
        float *input = l.input + img_index*l.inputs; // 定位到img
        float *output = l.output + img_index*l.outputs + box_index*4; // 定位到box
        float *truth = l.truth + img_index*grids*grids*7;
        float *conf = truth + box_index*7;
        if (conf[0] == 1.0){
            float *px = input+21*grids*grids+box_index;
            float *py = input+22*grids*grids+box_index;
            float *pw = input+23*grids*grids+box_index;
            float *ph = input+24*grids*grids+box_index;
            float *pc = input+box_index;

            float *tt = truth + box_index*7 + 1;
            float *tx = truth + box_index*7 + 2;
            float *ty = truth + box_index*7 + 3;
            float *tw = truth + box_index*7 + 4;
            float *th = truth + box_index*7 + 5;
            float *box_scale_weight = truth + box_index*7 + 6;

            float conf_loss = conf_msewithlogitsloss(pc[0], 1);
            float txty_loss = xy_bcewithlogitsloss(px[0], py[0], tx[0], ty[0], box_scale_weight[0]);
            float twth_loss = wh_mseloss(pw[0], ph[0], tw[0], th[0], box_scale_weight[0]);
            float clas_loss = class_crossentropy(input+grids*grids, tt[0], grids, grids, 20, box_index);
            float bbox_loss = txty_loss + twth_loss;
            output[0] = conf_loss;
            output[1] = clas_loss;
            output[2] = bbox_loss;
            output[3] = conf_loss + clas_loss + bbox_loss;
        } else {
            float *pc = input+box_index;
            float conf_loss = conf_msewithlogitsloss(pc[0], 0);
            output[0] = conf_loss;
            output[3] = conf_loss;
        }
    }
    yolo_loss(num, grids, l.output);
    sum_cpu(l.output, l.outputs*num, l.loss);
    multy_cpu(l.loss, 1, (float)1/(num*2), 1);
}

void backward_yolo_layer(Layer l, int num, float *n_delta)
{
    int grids = l.ksize / l.stride;
    for (int index = 0; index < num*grids*grids; ++index){
        int img_index = index / (grids*grids);
        int box_index = index % (grids*grids);
        float *input = l.input + img_index*l.inputs;
        float *delta = l.delta + img_index*l.inputs;
        float *truth = l.truth + img_index*grids*grids*7;
        float *conf = truth + box_index*7;
        if (conf[0] == 1.0){
            float *px = input+21*grids*grids+box_index;
            float *py = input+22*grids*grids+box_index;
            float *pw = input+23*grids*grids+box_index;
            float *ph = input+24*grids*grids+box_index;
            float *pc = input+box_index;

            float *tt = truth + box_index*7 + 1;
            float *tx = truth + box_index*7 + 2;
            float *ty = truth + box_index*7 + 3;
            float *tw = truth + box_index*7 + 4;
            float *th = truth + box_index*7 + 5;
            float *box_scale_weight = truth + box_index*7 + 6;

            conf_msewithlogitsloss_gradient(pc[0], 1, delta+box_index);
            xy_bcewithlogitsloss_gradient(px[0], py[0], tx[0], ty[0], box_scale_weight[0], delta+21*grids*grids+box_index, delta+22*grids*grids+box_index);
            wh_mseloss_gradient(pw[0], ph[0], tw[0], th[0], box_scale_weight[0], delta+23*grids*grids+box_index, delta+24*grids*grids+box_index);
            class_crossentropy_gradient(input+grids*grids, tt[0], grids, grids, 20, box_index, delta+grids*grids);
        } else {
            float *pc = input+box_index;
            conf_msewithlogitsloss_gradient(pc[0], 0, delta+box_index);
        }
    }
}

void zerograd_yolo_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
}
