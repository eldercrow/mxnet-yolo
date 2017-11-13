import mxnet as mx

from layer.anchor_box_layer import *
from layer.yolo_target_layer import *
from layer.dummy_layer import *
from layer.focal_loss_layer import *
from layer.smoothed_focal_loss_layer import *
# from layer.multibox_detection_layer import *
from layer.roi_transform_layer import *
from layer.iou_loss_layer import *
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


def get_preds(body, num_classes, use_global_stats):
    #
    anchor_shapes = cfg['anchor_shapes']
    num_anchor = len(anchor_shapes) / 2

    num_classes += 1

    # class prediction
    cls_preds = depthwise_unit(body, '_cls_pred',
            nf_dw=2048, nf_sep=0, kernel=(5, 5), pad=(2, 2),
            use_global_stats=use_global_stats)

    cls_pred_conv_bias = mx.sym.var(name='cls_pred_conv_bias',
            init=FocalBiasInit(num_classes, 0.01))
    cls_preds = mx.sym.Convolution(cls_preds, name='cls_pred_conv', bias=cls_pred_conv_bias,
            num_filter=num_anchor*num_classes, kernel=(1, 1), pad=(0, 0))

    # bb and iou prediction
    loc_preds = depthwise_unit(body, '_loc_pred',
            nf_dw=2048, nf_sep=0, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    loc_preds = mx.sym.Convolution(loc_preds, name='loc_pred_conv',
            num_filter=num_anchor*4, kernel=(1, 1), pad=(0, 0))

    anchor_boxes = mx.sym.Custom(body, op_type='anchor_box',
            name='yolo_anchors', shapes=anchor_shapes) # (1, n_anchor, 4)

    # reshape everything
    cls_preds = mx.sym.transpose(cls_preds, (0, 2, 3, 1))
    cls_preds = mx.sym.reshape(cls_preds, (0, -1, num_classes))
    cls_preds = mx.sym.transpose(cls_preds, (0, 2, 1)) # (n_batch, n_class, n_anchor)
    cls_probs = mx.sym.SoftmaxActivation(cls_preds, mode='channel')

    loc_preds = mx.sym.transpose(loc_preds, (0, 2, 3, 1))
    loc_preds = mx.sym.reshape(loc_preds, (0, -1, 4)) # (n_batch, n_anchor, 4)

    return cls_preds, cls_probs, loc_preds, anchor_boxes


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
    # use_focal_loss = cfg.train['use_focal_loss']
    # use_smooth_ce = cfg.train['use_smooth_ce']

    label = mx.sym.Variable('yolo_output_label')

    use_global_stats = False
    if 'use_global_stats' in kwargs:
        use_global_stats = kwargs['use_global_stats']
    else:
        kwargs['use_global_stats'] = False

    # sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    body = import_module(network).get_symbol(num_classes, **kwargs)

    cls_preds, cls_probs, loc_preds, anchor_boxes = \
            get_preds(body, num_classes, use_global_stats)

    # get target GT label
    th_small = 0.01 if not 'th_small' in kwargs else kwargs['th_small']
    tmp = mx.sym.Custom(*[anchor_boxes, label, cls_probs], name='yolo_target',
            op_type='yolo_target', th_small=th_small)
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    gamma = cfg.train['focal_loss_gamma']
    alpha = cfg.train['focal_loss_alpha']
    if not cfg.train['use_smooth_ce']:
        cls_preds, cls_loss = mx.sym.Custom(cls_preds, cls_probs, cls_target,
                op_type='focal_loss', name='cls_preds',
                gamma=gamma, alpha=alpha, normalize=True)
    else:
        th_prob = cfg.train['smooth_ce_th']
        w_reg = cfg.train['smooth_ce_lambda'] * float(num_classes)

        var_th_prob = mx.sym.var(name='th_prob_sce', shape=(1,), dtype=np.float32, \
                init=mx.init.Constant(np.log(th_prob)))
        var_th_prob = mx.sym.exp(var_th_prob)
        cls_preds, cls_loss = mx.sym.Custom(cls_preds, cls_probs, cls_target, var_th_prob,
                op_type='smoothed_focal_loss', name='cls_preds',
                gamma=gamma, alpha=alpha, th_prob=th_prob, w_reg=w_reg, normalize=True)

    # IOU loss
    loc_preds_det, _ = mx.symbol.Custom(loc_preds, anchor_boxes,
            name='roi_transform', op_type='roi_transform',
            variances=(0.1, 0.1, 0.2, 0.2))
    loc_loss, _, _, _ = mx.symbol.Custom(loc_preds_det, loc_target, loc_target_mask,
            name='iou_loss', op_type='iou_loss')
    # loc_loss = mx.sym.MakeLoss(loc_loss, name='loc_loss')

    # monitoring training status
    cls_label = mx.sym.BlockGrad(cls_target, name="cls_label")
    loc_label = mx.sym.BlockGrad(loc_target_mask, name='loc_label')

    loc_preds = mx.sym.reshape(loc_preds, (0, -1))
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_preds, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = [cls_loss, loc_loss, cls_label, loc_label, det]
    if cfg.train['use_smooth_ce']:
        out.append(mx.sym.BlockGrad(var_th_prob))
    return mx.sym.Group(out)


def get_symbol(network, num_classes,
        nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD
    """
    use_global_stats = True
    kwargs['use_global_stats'] = True

    body = import_module(network).get_symbol(num_classes, **kwargs)

    cls_preds, cls_probs, loc_preds, anchor_boxes = \
            get_preds(body, num_classes, use_global_stats)

    loc_preds_det = mx.sym.reshape(loc_preds, (0, -1))
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_probs, loc_preds_det, anchor_boxes], \
            name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
            variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk, clip=False)
    return out
