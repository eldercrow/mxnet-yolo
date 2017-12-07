import mxnet as mx

from layer.anchor_box_layer import *
from layer.yolo_target_layer import *
from layer.dummy_layer import *
from layer.focal_loss_layer import *
from layer.rpn_focal_loss_layer import *
from layer.smoothed_focal_loss_layer import *
from layer.rpn_smoothed_focal_loss_layer import *
from layer.multibox_detection_layer import *
from layer.roi_transform_layer import *
from layer.iou_loss_layer import *
from layer.log_wing_loss import *
from layer.merge_rpn_cls_layer import *
from config.config import cfg

from symbol.symbol_mobilenet import depthwise_unit


@mx.init.register
class FocalBiasInit(mx.init.Initializer):
    '''
    Initialize bias according to Focal Loss.
    '''
    def __init__(self, num_classes, pi=0.01):
        super(FocalBiasInit, self).__init__(num_classes=num_classes, pi=pi)
        self._num_classes = num_classes
        self._pi = pi

    def _init_weight(self, _, arr):
        data = np.full((arr.size,), -np.log((1.0 - self._pi) / self._pi))
        data = np.reshape(data, (-1, self._num_classes))
        data[:, 0] = 0
        arr[:] = data.ravel()


def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)


def get_preds(body_rpn, body_loc, body, num_classes, use_global_stats):
    #
    anchor_shapes = cfg['anchor_shapes']
    num_anchor = len(anchor_shapes) / 2

    num_classes += 1

    # rpn (objectness) prediction
    rpn_preds = depthwise_unit(body_rpn, 'rpn_pred_',
            nf_dw=256, nf_sep=0, kernel=(3, 3), pad=(1, 1),
            no_act=True, use_global_stats=use_global_stats)

    rpn_pred_conv_bias = mx.sym.var(name='rpn_pred_conv_bias',
            init=FocalBiasInit(2, 0.01))
    rpn_preds = mx.sym.Convolution(rpn_preds, name='rpn_pred_conv', bias=rpn_pred_conv_bias,
            num_filter=num_anchor*2, kernel=(1, 1), pad=(0, 0))

    # class prediction
    cls_preds = depthwise_unit(body, 'cls_pred_',
            nf_dw=1024, nf_sep=0, kernel=(5, 5), pad=(2, 2),
            no_act=True, use_global_stats=use_global_stats)

    cls_pred_conv_bias = mx.sym.var(name='cls_pred_conv_bias',
            init=FocalBiasInit(num_classes, 0.01))
    cls_preds = mx.sym.Convolution(cls_preds, name='cls_pred_conv', bias=cls_pred_conv_bias,
            num_filter=num_anchor*num_classes, kernel=(1, 1), pad=(0, 0))

    # bb and iou prediction
    loc_preds = depthwise_unit(body_loc, 'loc_pred_',
            nf_dw=256, nf_sep=0, kernel=(3, 3), pad=(1, 1),
            no_act=True, use_global_stats=use_global_stats)
    loc_preds = mx.sym.Convolution(loc_preds, name='loc_pred_conv',
            num_filter=num_anchor*4, kernel=(1, 1), pad=(0, 0))

    anchor_boxes = mx.sym.Custom(body, op_type='anchor_box',
            name='yolo_anchors', shapes=anchor_shapes) # (1, n_anchor, 4)

    # reshape everything
    rpn_preds = mx.sym.transpose(rpn_preds, (0, 2, 3, 1))
    rpn_preds = mx.sym.reshape(rpn_preds, (0, -1, 2))
    rpn_preds = mx.sym.transpose(rpn_preds, (0, 2, 1)) # (n_batch, n_class, n_anchor)
    rpn_probs = mx.sym.SoftmaxActivation(rpn_preds, mode='channel')

    cls_preds = mx.sym.transpose(cls_preds, (0, 2, 3, 1))
    cls_preds = mx.sym.reshape(cls_preds, (0, -1, num_classes))
    cls_preds = mx.sym.transpose(cls_preds, (0, 2, 1)) # (n_batch, n_class, n_anchor)
    cls_probs = mx.sym.SoftmaxActivation(cls_preds, mode='channel')

    loc_preds = mx.sym.transpose(loc_preds, (0, 2, 3, 1))
    loc_preds = mx.sym.reshape(loc_preds, (0, -1, 4)) # (n_batch, n_anchor, 4)

    return rpn_preds, rpn_probs, cls_preds, cls_probs, loc_preds, anchor_boxes


def get_symbol_train(network, num_classes,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_name : str
        feature extraction layer name
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('yolo_output_label')

    use_global_stats = False
    if 'use_global_stats' in kwargs:
        use_global_stats = kwargs['use_global_stats']
    else:
        kwargs['use_global_stats'] = False

    # sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    body_rpn, body_loc, body = import_module(network).get_symbol(num_classes, **kwargs)

    rpn_preds, rpn_probs, cls_preds, cls_probs, loc_preds, anchor_boxes = \
            get_preds(body_rpn, body_loc, body, num_classes, use_global_stats)

    # get target GT label
    tmp = mx.sym.Custom(*[anchor_boxes, label, cls_probs], name='yolo_target',
            op_type='yolo_target')
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]
    rpn_target = tmp[3]

    gamma = cfg.train['focal_loss_gamma']
    alpha = cfg.train['focal_loss_alpha']
    alpha_rpn = cfg.train['focal_loss_alpha_rpn']
    if not cfg.train['use_smooth_ce']:
        cls_preds, cls_loss = mx.sym.Custom(cls_preds, cls_probs, cls_target, rpn_preds, rpn_probs,
                op_type='rpn_focal_loss', name='cls_loss',
                gamma=gamma, alpha=alpha, normalize=True)
        rpn_preds, rpn_loss = mx.sym.Custom(rpn_preds, rpn_probs, rpn_target,
                op_type='focal_loss', name='rpn_loss',
                gamma=gamma, alpha=alpha_rpn, normalize=True)
    else:
        th_prob = cfg.train['smooth_ce_th']
        w_reg_cls = cfg.train['smooth_ce_lambda'] * float(num_classes+1)
        w_reg_rpn = cfg.train['smooth_ce_lambda'] * 2

        th_sce_cls = mx.sym.var(name='th_sce_cls', shape=(1,), dtype=np.float32, \
                init=mx.init.Constant(0.0)) #np.log(th_prob)))
        th_sce_cls = mx.sym.sigmoid(th_sce_cls)
        cls_preds, cls_loss = mx.sym.Custom( \
                cls_preds, cls_probs, cls_target, rpn_preds, rpn_probs, th_sce_cls, \
                op_type='rpn_smoothed_focal_loss', name='cls_loss', \
                gamma=gamma, alpha=alpha, normalize=True) #th_prob=th_prob, w_reg=w_reg_cls, normalize=True)

        th_sce_rpn = mx.sym.var(name='th_sce_rpn', shape=(1,), dtype=np.float32, \
                init=mx.init.Constant(0.0)) #np.log(th_prob)))
        th_sce_rpn = mx.sym.sigmoid(th_sce_rpn)
        rpn_preds, rpn_loss = mx.sym.Custom(rpn_preds, rpn_probs, rpn_target, th_sce_rpn,
                op_type='smoothed_focal_loss', name='rpn_loss',
                gamma=gamma, alpha=alpha_rpn, normalize=True) #th_prob=th_prob, w_reg=w_reg_rpn, normalize=True)

    # IOU loss
    loc_diff = (loc_preds - loc_target) * loc_target_mask
    # loc_loss = mx.sym.smooth_l1(loc_diff, name='log_wing_loss', scalar=1.0)
    loc_loss = mx.symbol.Custom(loc_diff, name='log_wing_loss', op_type='log_wing_loss')
    loc_loss = mx.sym.MakeLoss(loc_loss, name='loc_loss', normalization='valid')
    # loc_preds_det, _ = mx.symbol.Custom(loc_preds, anchor_boxes,
    #         name='roi_transform', op_type='roi_transform',
    #         variances=(0.1, 0.1, 0.2, 0.2))
    # loc_loss, _, _, _ = mx.symbol.Custom(loc_preds_det, loc_target, loc_target_mask,
    #         name='iou_loss', op_type='iou_loss')
    # loc_loss = mx.sym.MakeLoss(loc_loss, name='loc_loss')

    # monitoring training status
    cls_label = mx.sym.BlockGrad(cls_target, name="cls_label")
    loc_label = mx.sym.BlockGrad(loc_target_mask, name='loc_label')

    loc_preds = mx.sym.reshape(loc_preds, (0, -1))

    cls_merged = mx.sym.Custom(cls_preds, rpn_preds,
            name='merge_rpn_cls', op_type='merge_rpn_cls')
    #
    # det = mx.symbol.Custom(cls_preds, loc_preds_det, anchor_boxes, rpn_preds, \
    #         name='detection', op_type='multibox_detection',
    #         th_nms=nms_thresh, nms_topk=nms_topk, has_rpn=True)
    #
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_merged, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    #
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = [cls_loss, loc_loss, cls_label, loc_label, det, rpn_loss]
    if cfg.train['use_smooth_ce']:
        out.append(mx.sym.BlockGrad(th_sce_cls))
    return mx.sym.Group(out)


def get_symbol(network, num_classes,
        nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD
    """
    use_global_stats = True
    kwargs['use_global_stats'] = True

    body_rpn, body_loc, body = import_module(network).get_symbol(num_classes, **kwargs)
    rpn_preds, rpn_probs, cls_preds, cls_probs, loc_preds, anchor_boxes = \
            get_preds(body_rpn, body_loc, body, num_classes, use_global_stats)

    # body_rpn, body = import_module(network).get_symbol(num_classes, **kwargs)
    # rpn_preds, rpn_probs, cls_preds, cls_probs, loc_preds, anchor_boxes = \
    #         get_preds(body_rpn, body, num_classes, use_global_stats)

    loc_preds_det = mx.sym.reshape(loc_preds, (0, -1))

    cls_merged = mx.sym.Custom(cls_probs, rpn_probs,
            name='merge_rpn_cls', op_type='merge_rpn_cls')

    out = mx.contrib.symbol.MultiBoxDetection(*[cls_merged, loc_preds_det, anchor_boxes], \
            name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
            variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk, clip=False)
    return out
