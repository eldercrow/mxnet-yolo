"""
Reference:
Redmon, Joseph, and Ali Farhadi. "YOLO9000: Better, Faster, Stronger."
"https://arxiv.org/pdf/1612.08242.pdf"
"""
import mxnet as mx
from symbol_mobilenet import get_symbol as get_mobilenet
from symbol_mobilenet import conv_bn_relu, depthwise_unit


def subpixel_upsample(data, ch, c, r):
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0))  # (bsize*ch*r*c, a, b)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, r * c, 0, 0))  # (bsize*ch, r*c, a, b)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1))  # (bsize*ch, b, a, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c))  # (bsize*ch, b, a*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3))  # (bsize*ch, a*r, b, c)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, ch, 0, -3))  # (bsize, ch, a*r, b*c)
    return X


def get_symbol(num_classes, use_global_stats):
    #
    bone = get_mobilenet(num_classes=num_classes, use_global_stats=use_global_stats)

    conv5_5 = bone.get_internals()['relu5_5_sep_output']

    conv6 = bone.get_internals()['relu6_sep_output']
    conv6_1 = mx.sym.UpSampling(conv6, name='6_1', scale=2,
            num_filter=1024, sample_type='bilinear')
    # conv6_2 = conv_bn_relu(conv6_1, '6_2',
    #         num_filter=1024, kernel=(1, 1), pad=(0, 0),
    #         use_global_stats=use_global_stats)

    concat6 = mx.sym.concat(conv5_5, conv6_1, name='concat_6')

    # rpn
    rpn_1 = conv_bn_relu(concat6, 'rpn_1',
            num_filter=256, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    # extra layers
    conv7_1 = conv_bn_relu(concat6, '7_1',
            num_filter=1024, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    conv8_1 = depthwise_unit(conv7_1, '8_1',
            nf_dw=1024, nf_sep=1024, kernel=(5, 5), pad=(2, 2),
            use_global_stats=use_global_stats)
    # conv8_2 = depthwise_unit(conv8_1, '8_2',
    #         nf_dw=1024, nf_sep=1024, kernel=(3, 3), pad=(1, 1),
    #         use_global_stats=use_global_stats)
    # conv7_3 = depthwise_unit(conv7_2, '7_3',
    #         nf_dw=1024, nf_sep=1024, kernel=(3, 3), pad=(1, 1),
    #         use_global_stats=use_global_stats)
    #
    # # re-organze conv5_5 and concat conv7_2
    # concat = mx.sym.concat(conv5_6, conv7_2)
    # # concat = conv7_2
    # conv8_1 = depthwise_unit(concat, '8_1',
    #         nf_dw=1024+2048, nf_sep=1024, kernel=(3, 3), pad=(1, 1),
    #         use_global_stats=use_global_stats)

    return rpn_1, conv8_1
    #
    # th_small = 0.04 if not 'th_small' in kwargs else kwargs['th_small']
    # cls_probs = mx.sym.SoftmaxActivation(cls_preds, mode='channel')
    # tmp = mx.sym.Custom(*[anchor_boxes, label, cls_probs], name='yolo_output',
    #         op_type='yolo_output', th_small=th_small)
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]
    #
    # gamma = cfg.train['focal_loss_gamma']
    # alpha = cfg.train['focal_loss_alpha']
    # cls_loss = mx.sym.Custom(cls_preds, cls_probs, cls_target, op_type='focal_loss', name='cls_loss',
    #         gamma=gamma, alpha=alpha, normalize=True)
    #
    # loc_diff = loc_target_mask * (loc_preds - loc_target)
    # loc_loss = mx.sym.MakeLoss(loc_diff*loc_diff, grad_scale=cfg.train['smoothl1_weight'], \
    #         normalization='valid', name="loc_loss")
    #
    # # monitoring training status
    # cls_label = mx.sym.BlockGrad(cls_target, name="cls_label")
    # loc_label = mx.sym.BlockGrad(loc_target_mask, name='loc_label')
    #
    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_loss, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_nms,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    #
    # # group output
    # out = [cls_loss, loc_loss, cls_label, loc_label, det]
    # return mx.sym.Group(out)

    # pred = mx.symbol.Convolution(data=conv8_1, name='conv_pred', kernel=(1, 1),
    #     num_filter=num_anchor * (num_classes + 4 + 1))
    #
    # out = mx.contrib.symbol.YoloOutput(data=pred, num_class=num_classes,
    #     num_anchor=num_anchor, object_grad_scale=5.0, background_grad_scale=1.0,
    #     coord_grad_scale=1.0, class_grad_scale=1.0, anchors=anchors,
    #     nms_topk=400, warmup_samples=12800, name='yolo_output')
    # return out
