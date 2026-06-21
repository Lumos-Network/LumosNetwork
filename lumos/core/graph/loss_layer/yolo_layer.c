#include "yolo_layer.h"

float compute_iou(int S, float xp, float yp, float wp, float hp, float xt, float yt, float wt, float ht)
{
    float cell_size = 1 / S;
    xp *= cell_size;
    yp *= cell_size;
    float px1 = xp - wp / 2;
    float py1 = yp - hp / 2;
    float px2 = xp + wp / 2;
    float py2 = yp + hp / 2;

    xt *= cell_size;
    yt *= cell_size;
    float tx1 = xt - wt / 2;
    float ty1 = yt - ht / 2;
    float tx2 = xt + wt / 2;
    float ty2 = yt + ht / 2;

    float max_x = (px1>tx1) ? px1:tx1;
    float max_y = (py1>ty1) ? py1:ty1;
    float min_x = (px2<tx2) ? px2:tx2;
    float min_y = (py2<ty2) ? py2:ty2;

    float inter_w = fabs(min_x-max_x);
    float inter_h = fabs(min_y-max_y);
    float inter_area = inter_w * inter_h;

    float parea = wp * hp;
    float tarea = wt * wp;
    float uarea = parea + tarea - inter_area + 1e-6;
    float iou = inter_area / uarea;
    return iou;
}

Layer *make_yolo_layer()
{
    Layer *l = malloc(sizeof(Layer));
    l->type = YOLO;

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

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = 1;
    l->outputs = 7*7;

    l->workspace_size = 0;

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "YOLO    Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

// 7*7*(5*2+20) [box1_x,box1_y,box1_w,box1_h,box1_conf,box2_x,box2_y,box2_w,box2_h,box2_conf,cls0~cls19]
void forward_yolo_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        float *input = l.input + i*l.inputs;
        float *output = l.output + i*l.outputs;
        float *truth = l.truth + i*7*7*30;
        for (int j = 0; j < 7*7; ++j){
            float *cell_p = input + j*30;
            float *cell_t = truth + j*30;
            float px1 = cell_p[0];
            float py1 = cell_p[1];
            float pw1 = cell_p[2];
            float ph1 = cell_p[3];
            float px2 = cell_p[5];
            float py2 = cell_p[6];
            float pw2 = cell_p[7];
            float ph2 = cell_p[8];
            float tx1 = cell_t[0];
            float ty1 = cell_t[1];
            float tw1 = cell_t[2];
            float th1 = cell_t[3];
            float tc1 = cell_t[4];
            float iou1 = compute_iou(7, px1, py1, pw1, ph1, tx1, ty1, tw1, th1);
            float iou2 = compute_iou(7, px2, py2, pw2, ph2, tx1, ty1, tw1, th1);
            int mask = (iou1>iou2) ? 0:1;
            float bx = (1-mask)*px1 + mask*px2;
            float by = (1-mask)*py1 + mask*py2;
            float bw = (1-mask)*pw1 + mask*pw2;
            float bh = (1-mask)*ph1 + mask*ph2;
            float bc = (1-mask)*cell_p[4] + mask*cell_p[9];
            bx = logistic_activate(bx);
            by = logistic_activate(by);
            bw = logistic_activate(bw);
            bh = logistic_activate(bh);
            float lossxy = powf(bx-tx1, 2) + powf(by-ty1, 2);
            float losswh = powf(sqrtf(bw)-sqrtf(tw1), 2) + powf(sqrtf(bh)-sqrtf(th1), 2);
            float loss_coord = 5 * tc1 * (lossxy + losswh);
            float loss_noobj = tc1*powf(bc-tc1, 2)+0.5*(1-tc1)*powf(bc-tc1, 2);
            float loss_class = 0;
            for (int k = 0; k < 20; ++k){
                loss_class += tc1*powf((cell_p[10+k]-cell_t[10+k]), 2);
            }
            output[j] = loss_coord + loss_noobj + loss_class;
        }
    }
    sum_cpu(l.output, l.outputs*num, l.loss);
    multy_cpu(l.loss, 1, (float)1/num, 1);
}

void backward_yolo_layer(Layer l, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        float *input = l.input + i*l.inputs;
        float *delta = l.delta + i*l.inputs;
        float *truth = l.truth + i*7*7*30;
        for (int j = 0; j < 7*7; ++j){
            float *cell_p = input + j*30;
            float *cell_t = truth + j*30;
            float px1 = cell_p[0];
            float py1 = cell_p[1];
            float pw1 = cell_p[2];
            float ph1 = cell_p[3];
            float px2 = cell_p[5];
            float py2 = cell_p[6];
            float pw2 = cell_p[7];
            float ph2 = cell_p[8];
            float tx1 = cell_t[0];
            float ty1 = cell_t[1];
            float tw1 = cell_t[2];
            float th1 = cell_t[3];
            float tc1 = cell_t[4];
            float iou1 = compute_iou(7, px1, py1, pw1, ph1, tx1, ty1, tw1, th1);
            float iou2 = compute_iou(7, px2, py2, pw2, ph2, tx1, ty1, tw1, th1);
            float *delta_mask = (iou1>iou2) ? (delta+j*30):(delta+j*30+5);
            int mask = (iou1>iou2) ? 0:1;
            float bx = (1-mask)*px1 + mask*px2;
            float by = (1-mask)*py1 + mask*py2;
            float bw = (1-mask)*pw1 + mask*pw2;
            float bh = (1-mask)*ph1 + mask*ph2;
            float bc = (1-mask)*cell_p[4] + mask*cell_p[9];
            float gbx = logistic_gradient(bx);
            float gby = logistic_gradient(by);
            float gbw = logistic_gradient(bw);
            float gbh = logistic_gradient(bh);
            bx = logistic_activate(bx);
            by = logistic_activate(by);
            bw = logistic_activate(bw);
            bh = logistic_activate(bh);
            delta_mask[0] = 5 * tc1 * 2*(bx-tx1) * gbx;
            delta_mask[1] = 5 * tc1 * 2*(by-ty1) * gby;
            delta_mask[2] = 5 * tc1 * (sqrtf(bw)-sqrtf(tw1)) / sqrtf(bw) * gbw;
            delta_mask[3] = 5 * tc1 * (sqrtf(bh)-sqrtf(th1)) / sqrtf(bh) * gbh;
            delta_mask[4] = 0.5 * tc1 * 2 * bc + (1-tc1) * 2 * bc;
            float *delta_class = delta+j*30+10;
            for (int k = 0; k < 20; ++k){
                delta_class[0] = tc1 * 2 * (cell_p[10+k]-cell_t[10+k]);
            }
        }
    }
}

void zerograd_yolo_layer(Layer l, int subdivision)
{
    fill_cpu(l.delta, subdivision*l.inputs, 0, 1);
}
