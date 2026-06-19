#include "yolo_layer_gpu.h"

__device__ float compute_iou_gpu(int S, float xp, float yp, float wp, float hp, float xt, float yt, float wt, float ht)
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

__global__ void yolo_kernel(int num, float *pretrained, int lp, float *targets, float *space, int ls)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num*7*7) return;
    int img_index = index / (7*7);
    int box_index = index % (7*7);
    float *input = pretrained + img_index*lp;
    float *output = space + img_index*ls;
    float *truth = targets + img_index*7*7*30;
    float *cell_p = input + box_index*30;
    float *cell_t = truth + box_index*30;
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
    float iou1 = compute_iou_gpu(7, px1, py1, pw1, ph1, tx1, ty1, tw1, th1);
    float iou2 = compute_iou_gpu(7, px2, py2, pw2, ph2, tx1, ty1, tw1, th1);
    int mask = (iou1>iou2) ? 0:1;
    float bx = (1-mask)*px1 + mask*px2;
    float by = (1-mask)*py1 + mask*py2;
    float bw = (1-mask)*pw1 + mask*pw2;
    float bh = (1-mask)*ph1 + mask*ph2;
    float bc = (1-mask)*cell_p[4] + mask*cell_p[9];
    float lossxy = powf(bx-tx1, 2) + powf(by-ty1, 2);
    float losswh = powf(sqrtf(bw)-sqrtf(tw1), 2) + powf(sqrtf(bh)-sqrtf(th1), 2);
    float loss_coord = 5 * tc1 * (lossxy + losswh);
    float loss_noobj = tc1*powf(bc-tc1, 2)+0.5*(1-tc1)*powf(bc-tc1, 2);
    float loss_class = 0;
    for (int k = 0; k < 20; ++k){
        loss_class += tc1*powf((cell_p[10+k]-cell_t[10+k]), 2);
    }
    output[box_index] = loss_coord + loss_noobj + loss_class;
}

__global__ void yolo_gradient_kernel(int num, float *pretrained, int lp, float *targets, float *space)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num*7*7) return;
    int img_index = index / (7*7);
    int box_index = index % (7*7);
    float *input = pretrained + img_index*lp;
    float *delta = space + img_index*lp;
    float *truth = targets + img_index*7*7*30;
    float *cell_p = input + box_index*30;
    float *cell_t = truth + box_index*30;
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
    float iou1 = compute_iou_gpu(7, px1, py1, pw1, ph1, tx1, ty1, tw1, th1);
    float iou2 = compute_iou_gpu(7, px2, py2, pw2, ph2, tx1, ty1, tw1, th1);
    float *delta_mask = (iou1>iou2) ? (delta+box_index*30):(delta+box_index*30+5);
    int mask = (iou1>iou2) ? 0:1;
    float bx = (1-mask)*px1 + mask*px2;
    float by = (1-mask)*py1 + mask*py2;
    float bw = (1-mask)*pw1 + mask*pw2;
    float bh = (1-mask)*ph1 + mask*ph2;
    float bc = (1-mask)*cell_p[4] + mask*cell_p[9];
    delta_mask[0] = 5 * tc1 * 2*(bx-tx1);
    delta_mask[1] = 5 * tc1 * 2*(by-ty1);
    delta_mask[2] = 5 * tc1 * (sqrtf(bw)-sqrtf(tw1)) / sqrtf(bw);
    delta_mask[3] = 5 * tc1 * (sqrtf(bh)-sqrtf(th1)) / sqrtf(bh);
    delta_mask[4] = 0.5 * tc1 * 2 * bc + (1-tc1) * 2 * bc;
    float *delta_class = delta+box_index*30+10;
    for (int k = 0; k < 20; ++k){
        delta_class[0] = tc1 * 2 * (cell_p[10+k]-cell_t[10+k]);
    }
}

void init_yolo_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = 1;
    l->outputs = 1;

    l->workspace_size = 0;

    cudaMalloc((void**)l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)l->delta, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "YOLO    Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_yolo_layer_gpu(Layer l, int num)
{
    int size = num*7*7;
    int block_num = (size + BLOCK-1) / BLOCK;
    yolo_kernel<<<block_num, BLOCK>>>(num, l.input, l.inputs, l.truth, l.output, l.outputs);
}

void backward_yolo_layer_gpu(Layer l, int num, float *n_delta)
{
    int size = num*7*7;
    int block_num = (size + BLOCK-1) / BLOCK;
    yolo_gradient_kernel<<<block_num, BLOCK>>>(num, l.input, l.inputs, l.truth, l.delta);
}

void zerograd_yolo_layer_gpu(Layer l, int subdivision)
{
    fill_gpu(l.delta, subdivision*l.inputs, 0, 1);
}
