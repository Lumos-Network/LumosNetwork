#include "yolov1.h"
// #include "cpu.h"

// #define MAX(a, b) ((a) > (b) ? (a) : (b))
// #define MIN(a, b) ((a) < (b) ? (a) : (b))

// float conf_msewithlogitsloss_gpu(float pred, float target)
// {
//     float input = 1 / (1 + exp(-pred));
//     if (input < 1e-4) input = 1e-4;
//     if (input > 1.0-(1e-4)) input = 1.0-(1e-4);
//     float pos_id = (target==1.0)?1.0:0;
//     float neg_id = (target==0.0)?1.0:0;

//     float pos_loss = pos_id * pow(input-target, 2);
//     float neg_loss = neg_id * pow(input, 2);

//     float conf_loss = 5.0*pos_loss + 1.0*neg_loss;
//     return conf_loss;
// }

// void conf_msewithlogitsloss_gradient_gpu(float pred, float target, float *space)
// {
//     float input = 1 / (1 + exp(-pred));
//     if (input < 1e-4) input = 1e-4;
//     if (input > 1.0-(1e-4)) input = 1.0-(1e-4);

//     float pos_id = (target==1.0)?1.0:0;
//     float neg_id = (target==0.0)?1.0:0;

//     float pos_delta = pos_id * 2*(input-target);
//     float neg_delta = neg_id * 2*input;

//     float conf_delta = 5.0*pos_delta + 1.0*neg_delta;
//     conf_delta *= (1-input)*input;
//     space[0] = conf_delta;
// }

// float xy_bcewithlogitsloss_gpu(float px, float py, float tx, float ty, float box_scale_weight)
// {
//     float x_loss = 0;
//     float y_loss = 0;
//     x_loss = -px*tx + log(1+exp(px));
//     y_loss = -py*ty + log(1+exp(py));
//     float xy_loss = (x_loss + y_loss)*box_scale_weight;
//     return xy_loss;
// }

// void xy_bcewithlogitsloss_gradient_gpu(float px, float py, float tx, float ty, float box_scale_weight, float *space)
// {
//     float x_delta = 0;
//     float y_delta = 0;
//     x_delta = 1 / (1 + exp(-px)) - tx;
//     y_delta = 1 / (1 + exp(-py)) - ty;
//     space[0] = x_delta*box_scale_weight;
//     space[1] = y_delta*box_scale_weight;
// }

// float wh_mseloss_gpu(float pw, float ph, float tw, float th, float box_scale_weight)
// {
//     float w_loss = powf(pw-tw, 2);
//     float h_loss = powf(ph-th, 2);
//     float wh_loss = (w_loss + h_loss)*box_scale_weight;
//     return wh_loss;
// }

// void wh_mseloss_gradient_gpu(float pw, float ph, float tw, float th, float box_scale_weight, float *space)
// {
//     float w_delta = 2*(pw-tw);
//     float h_delta = 2*(ph-th);
//     space[0] = w_delta*box_scale_weight;
//     space[1] = h_delta*box_scale_weight;
// }

// float class_crossentropy_kernel(float *data, float truth, int w, int h, int c, float *scale, int ignore, int index)
// {
//     float scale_x = 1;
//     int target = (int)truth;
//     if (ignore != -1){
//         if (target == ignore){
//             return 0.0;
//         }
//     }
//     if (scale != NULL){
//         scale_x = scale[target];
//     }
//     float max_val = -INFINITY;
//     float sum_exp = 0;
//     for (int i = 0; i < c; ++i){
//         max_val = MAX(max_val, data[i*w*h+index]);
//     }
//     for (int i = 0; i < c; ++i){
//         sum_exp += expf(data[i*w*h+index]-max_val);
//     }
//     float loss = (-data[target*w*h+index]+max_val+log(sum_exp))*scale_x;
//     return loss;
// }

// void class_crossentropy_gradient_kernel(float *data, float truth, int w, int h, int c, float *scale, int ignore, int index, float *space)
// {
//     float scale_x = 1;
//     int target = (int)truth;
//     if (ignore != -1){
//         if (target == ignore){
//             for (int i = 0; i < c; ++i){
//                 space[i*w*h+index] = 0.0;
//             }
//             return;
//         }
//     }
//     if (scale != NULL){
//         scale_x = scale[target];
//     }
//     float max_val = -INFINITY;
//     float sum_exp = 0;
//     for (int i = 0; i < c; ++i){
//         max_val = MAX(max_val, data[i*w*h+index]);
//     }
//     for (int i = 0; i < c; ++i){
//         space[i*w*h+index] = expf(data[i*w*h+index]-max_val);
//         sum_exp += space[i*w*h+index];
//     }
//     for (int i = 0; i < c; ++i){
//         if (i == target) space[i*w*h+index] = (space[i*w*h+index]/sum_exp-1)/(w*h)*scale_x;
//         else space[i*w*h+index] = (space[i*w*h+index]/sum_exp)/(w*h)*scale_x;
//     }
// }

int main()
{
    // FILE *fpconf = fopen("./backup/pred_conf", "rb");
    // FILE *fpclas = fopen("./backup/pred_class", "rb");
    // FILE *fpxy = fopen("./backup/pred_txty", "rb");
    // FILE *fpwh = fopen("./backup/pred_twth", "rb");
    // float *pred_conf = malloc(4*169*sizeof(float));
    // float *pred_cls = malloc(4*20*169*sizeof(float));
    // float *pred_txty = malloc(4*169*2*sizeof(float));
    // float *pred_twth = malloc(4*169*2*sizeof(float));
    // fread(pred_conf, sizeof(float), 4*169, fpconf);
    // fread(pred_cls, sizeof(float), 4*20*169, fpclas);
    // fread(pred_txty, sizeof(float), 4*169*2, fpxy);
    // fread(pred_twth, sizeof(float), 4*169*2, fpwh);
    // fclose(fpconf);
    // fclose(fpclas);
    // fclose(fpxy);
    // fclose(fpwh);

    // FILE *ftconf = fopen("./backup/target_conf", "rb");
    // FILE *ftclas = fopen("./backup/target_class", "rb");
    // FILE *ftxy = fopen("./backup/target_txty", "rb");
    // FILE *ftwh = fopen("./backup/target_twth", "rb");
    // FILE *ftbox = fopen("./backup/target_box", "rb");
    // float *gt_obj = malloc(4*169*sizeof(float));
    // float *gt_cls = malloc(4*169*sizeof(float));
    // float *gt_txty = malloc(4*169*2*sizeof(float));
    // float *gt_twth = malloc(4*169*2*sizeof(float));
    // float *gt_box_scale_weight = malloc(4*169*sizeof(float));
    // fread(gt_obj, sizeof(float), 4*169, ftconf);
    // fread(gt_cls, sizeof(float), 4*169, ftclas);
    // fread(gt_txty, sizeof(float), 4*169*2, ftxy);
    // fread(gt_twth, sizeof(float), 4*169*2, ftwh);
    // fread(gt_box_scale_weight, sizeof(float), 4*169, ftbox);
    // fclose(ftconf);
    // fclose(ftclas);
    // fclose(ftxy);
    // fclose(ftwh);
    // fclose(ftbox);

    // float total_class = 0;
    // float conf_loss = 0;
    // for (int i = 0; i < 4*169; ++i){
    //     conf_loss += conf_msewithlogitsloss_gpu(pred_conf[i], gt_obj[i]);
    // }
    // conf_loss /= 4;
    // printf("conf loss:%f\n", conf_loss);

    // float cls_loss = 0;
    // for (int i = 0; i < 4*169; ++i){
    //     if (gt_obj[i] == 1.0){
    //         int img_index = i / 169;
    //         int box_index = i % 169;
    //         cls_loss += class_crossentropy_kernel(pred_cls+img_index*169*20, gt_cls[img_index*169+box_index], 13, 13, 20, NULL, -1, box_index);
    //     }
    // }
    // cls_loss /= 4;
    // printf("class loss:%f\n", cls_loss);

    // float xy_loss = 0;
    // for (int i = 0; i < 4*169; ++i){
    //     if (gt_obj[i] == 1.0){
    //         int img_index = i / 169;
    //         int box_index = i % 169;
    //         float *px = pred_txty + img_index * 169*2 + box_index*2;
    //         float *py = pred_txty + img_index * 169*2 + box_index*2 + 1;
    //         float *tx = gt_txty + img_index * 169*2 + box_index*2;
    //         float *ty = gt_txty + img_index * 169*2 + box_index*2 + 1;
    //         float *box = gt_box_scale_weight + img_index * 169 + box_index;
    //         xy_loss += xy_bcewithlogitsloss_gpu(px[0], py[0], tx[0], ty[0], box[0]);
    //     }
    // }
    // xy_loss /= 4;
    // printf("xy loss:%f\n", xy_loss);

    // float wh_loss = 0;
    // for (int i = 0; i < 4*169; ++i){
    //     if (gt_obj[i] == 1.0){
    //         int img_index = i / 169;
    //         int box_index = i % 169;
    //         float *pw = pred_twth + img_index * 169*2 + box_index*2;
    //         float *ph = pred_twth + img_index * 169*2 + box_index*2 + 1;
    //         float *tw = gt_twth + img_index * 169*2 + box_index*2;
    //         float *th = gt_twth + img_index * 169*2 + box_index*2 + 1;
    //         float *box = gt_box_scale_weight + img_index * 169 + box_index;
    //         wh_loss += wh_mseloss_gpu(pw[0], ph[0], tw[0], th[0], box[0]);
    //     }
    // }
    // wh_loss /= 4;
    // printf("wh loss:%f\n", wh_loss);

    // total_class = conf_loss + cls_loss + xy_loss + wh_loss;
    // total_class /= 8;
    // printf("total loss:%f\n", total_class);

    // float *delta = malloc(4*169*25*sizeof(float));
    // fill_cpu(delta, 4*169*25, 0, 1);

    // for (int i = 0; i < 4*169; ++i){
    //     int img_index = i / 169;
    //     int box_index = i % 169;
    //     conf_msewithlogitsloss_gradient_gpu(pred_conf[i], gt_obj[i], delta+img_index*169*25+box_index);
    // }

    // for (int i = 0; i < 4*169; ++i){
    //     int img_index = i / 169;
    //     int box_index = i % 169;
    //     if (gt_obj[i] == 1.0){
    //         class_crossentropy_gradient_kernel(pred_cls+img_index*169*20, gt_cls[img_index*169+box_index], 13, 13, 20, NULL, -1, box_index, delta+img_index*169*25+169);
    //     }
    // }

    // for (int i = 0; i < 4*169; ++i){
    //     if (gt_obj[i] == 1.0){
    //         int img_index = i / 169;
    //         int box_index = i % 169;
    //         float *pw = pred_twth + img_index * 169*2 + box_index*2;
    //         float *ph = pred_twth + img_index * 169*2 + box_index*2 + 1;
    //         float *tw = gt_twth + img_index * 169*2 + box_index*2;
    //         float *th = gt_twth + img_index * 169*2 + box_index*2 + 1;
    //         float *box = gt_box_scale_weight + img_index * 169 + box_index;
    //         xy_bcewithlogitsloss_gradient_gpu(pw[0], ph[0], tw[0], th[0], box[0], delta+img_index*169*25+)
    //     }
    // }
    yolov1("gpu", "./backup/yolo-20");
}