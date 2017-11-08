"""
Reference:
Redmon, Joseph, and Ali Farhadi. "YOLO9000: Better, Faster, Stronger."
"https://arxiv.org/pdf/1612.08242.pdf"
"""
import mxnet as mx
from symbol_mobilenet import get_symbol as get_mobilenet
from symbol_mobilenet import depthwise_unit, subpixel_downsample

def get_symbol(num_classes=20, nms_thresh=0.5, force_nms=False, **kwargs):
    #
    if 'use_global_stats' not in kwargs:
        kwargs['use_global_stats'] = False

    bone = get_mobilenet(num_classes=num_classes, **kwargs)
    conv5_5 = bone.get_internals()['conv5_5_output']
    conv6 = bone.get_internals()['conv6_output']
    # anchors
    anchors = [
               1.3221, 1.73145,
               3.19275, 4.00944,
               5.05587, 8.09892,
               9.47112, 4.84053,
               11.2364, 10.0071]
    num_anchor = len(anchors) // 2

    # extra layers
    conv7_1 = depthwise_unit(conv6, '7_1',
            nf_dw=1024, nf_sep=1024, kernel=(3, 3), pad=(1, 1),
            use_global_stats=kwargs['use_global_stats'])
    conv7_2 = depthwise_unit(conv7_1, '7_2',
            nf_dw=1024, nf_sep=1024, kernel=(3, 3), pad=(1, 1),
            use_global_stats=kwargs['use_global_stats'])

    # re-organze conv5_5 and concat conv7_2
    conv5_6 = subpixel_downsample(conv5_5, 512, 2, 2)
    concat = mx.sym.concat(conv5_6, conv7_2)
    # concat = conv7_2
    conv8_1 = depthwise_unit(concat, '8_1',
            nf_dw=1024+2048, nf_sep=1024, kernel=(3, 3), pad=(1, 1),
            use_global_stats=kwargs['use_global_stats'])

    loc_preds = depthwise_unit(conv8_1, '_loc_pred',
            nf_dw=1024, nf_sep=5, kernel=(3, 3), pad=(1, 1),
            use_global_stats=kwargs['use_global_stats'])
    cls_preds = depthwise_unit(conv8_1, '_cls_pred',
            nf_dw=1024, nf_sep=num_classes, kernel=(5, 5), pad=(2, 2),
            use_global_stats=kwargs['use_global_stats'])
    anchor_boxes = mx.sym.Custom(conv8_1, op_type='multibox_prior',
            name='yolo_anchors', shapes=anchors)

    th_small = 0.04 if not 'th_small' in kwargs else kwargs['th_small']
    cls_probs = mx.sym.SoftmaxActivation(cls_preds, mode='channel')
    tmp = mx.sym.Custom(*[anchor_boxes, label, cls_probs], name='yolo_output',
            op_type='yolo_output', th_small=th_small)
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    gamma = cfg.train['focal_loss_gamma']
    alpha = cfg.train['focal_loss_alpha']
    cls_loss = mx.sym.Custom(cls_preds, cls_probs, cls_target, op_type='focal_loss', name='cls_loss',
            gamma=gamma, alpha=alpha, normalize=True)

    loc_diff = loc_target_mask * (loc_preds - loc_target)
    loc_loss = mx.sym.MakeLoss(loc_diff*loc_diff, grad_scale=cfg.train['smoothl1_weight'], \
            normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.sym.BlockGrad(cls_target, name="cls_label")
    loc_label = mx.sym.BlockGrad(loc_target_mask, name='loc_label')

    det = mx.contrib.symbol.MultiBoxDetection(*[cls_loss, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = [cls_loss, loc_loss, cls_label, loc_label, det, match_info]
    return mx.sym.Group(out)

    # pred = mx.symbol.Convolution(data=conv8_1, name='conv_pred', kernel=(1, 1),
    #     num_filter=num_anchor * (num_classes + 4 + 1))
    #
    # out = mx.contrib.symbol.YoloOutput(data=pred, num_class=num_classes,
    #     num_anchor=num_anchor, object_grad_scale=5.0, background_grad_scale=1.0,
    #     coord_grad_scale=1.0, class_grad_scale=1.0, anchors=anchors,
    #     nms_topk=400, warmup_samples=12800, name='yolo_output')
    # return out
