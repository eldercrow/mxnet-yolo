# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"

import mxnet as mx
import numpy as np
import logging
from ast import literal_eval

class YoloTarget(mx.operator.CustomOp):
    '''
    Python (inexact) implementation of yolo output layer.
    '''
    def __init__(self, th_iou, th_iou_neg, th_iou_pass):
        #
        super(YoloTarget, self).__init__()
        self.th_iou = th_iou
        self.th_iou_neg = th_iou_neg
        self.th_iou_pass = th_iou_pass

        # precompute nms candidates
        self.anchors = None
        self.anchors_t = None
        self.area_anchors_t = None

    def forward(self, is_train, req, in_data, out_data, aux):
        #
        # inputs:  ['anchors', 'label', 'probs_cls']
        # outputs: ['target_reg', 'mask_reg', 'target_cls']
        n_batch, nch, n_anchors = in_data[2].shape

        labels_all = in_data[1].asnumpy().astype(np.float32) # (batch, num_label, 6)
        labels_all = labels_all[:, :, :5] # last one is difficulty, which I won't use.
        max_cids = mx.nd.argmax(in_data[2], axis=1).asnumpy().astype(int)

        # precompute some data for IOU computation
        if self.anchors_t is None:
            self.anchors = np.reshape(in_data[0].asnumpy(), (-1, 4)) # (n_anchor, 4)
            self.anchors_t = mx.nd.transpose(mx.nd.reshape(in_data[0].copy(), shape=(-1, 4)), (1, 0))
            self.area_anchors_t = \
                    (self.anchors_t[2] - self.anchors_t[0]) * (self.anchors_t[3] - self.anchors_t[1])

        # numpy arrays for outputs of the layer
        target_reg = np.zeros((n_batch, n_anchors, 4), dtype=np.float32)
        mask_reg = np.zeros_like(target_reg)
        # I will use focal loss, so basically everything is negative.
        target_cls = np.zeros((n_batch, 1, n_anchors), dtype=np.float32)

        # mark per-batch positive and ignore samples
        for i in range(n_batch):
            target_cls[i][0], target_reg[i], mask_reg[i] = self._forward_batch_pos( \
                    labels_all[i], max_cids[i], \
                    target_cls[i][0], target_reg[i], mask_reg[i])

        target_reg = np.reshape(target_reg, (n_batch, -1, 4))
        mask_reg = np.reshape(mask_reg, (n_batch, -1, 4))

        self.assign(out_data[0], req[0], mx.nd.array(target_reg, ctx=in_data[2].context))
        self.assign(out_data[1], req[1], mx.nd.array(mask_reg, ctx=in_data[2].context))
        self.assign(out_data[2], req[2], mx.nd.array(target_cls, ctx=in_data[2].context))
        self.assign(out_data[3], req[3], mx.nd.array(np.minimum(1.0, target_cls), ctx=in_data[2].context))

    def _forward_batch_pos(self, labels, max_cids, target_cls, target_reg, mask_reg):
        '''
        labels: (n_label, 5)
        max_cids: (n_anchor, )
        target_cls: (n_anchor, )
        target_reg: (n_anchor, 4)
        mask_reg: (n_anchor, 4)
        '''
        n_anchors = self.anchors_t.shape[1]

        labels = _get_valid_labels(labels)
        max_iou = np.zeros(n_anchors, dtype=np.float32)

        for i, label in enumerate(labels):
            gt_cls = int(label[0]) + 1
            #
            lsq = _autofit_ratio(label[1:], max_ratio=3.0)
            #
            iou = _compute_iou(lsq, self.anchors_t, self.area_anchors_t)

            # skip already occupied ones
            iou_mask = iou > max_iou
            max_iou = np.maximum(iou, max_iou)
            if label[0] == -1:
                continue
            gt_sz = np.maximum(label[3]-label[1], label[4]-label[2])

            # positive and regression samples
            pidx = np.where(np.logical_and(iou_mask, iou > self.th_iou))[0]
            ridx = np.where(np.logical_and(iou_mask, iou > self.th_iou_neg))[0]

            if len(pidx) > 5:
                pidx = np.random.choice(pidx, 5, replace=False)
            elif len(pidx) < 3:
                # TEST
                iou_v = _compute_iou(_adjust_ratio(lsq, 2.0), self.anchors_t, self.area_anchors_t)
                iou_h = _compute_iou(_adjust_ratio(lsq, 0.5), self.anchors_t, self.area_anchors_t)

                iou_t = np.maximum(np.maximum(iou, iou_v), iou_h)
                if np.max(iou_t) < self.th_iou_pass:
                    continue

                sidx = np.argpartition(iou_t, iou_t.size - 5)
                pidx = sidx[-5:]
                pidx = pidx[np.where(iou_t[pidx] > self.th_iou_pass)[0]]
                # ridx = sidx[-5:]
                # pidx = pidx[np.where(iou_t[ridx] > self.th_iou_pass)[0]]

            # map ridx first, and then pidx
            ridx = ridx[target_cls[ridx] == 0]
            target_cls[ridx] = -1
            if len(pidx) > 0:
                target_cls[pidx] = gt_cls
                rt, rm = _compute_loc_target(label[1:], self.anchors[pidx, :])
                target_reg[pidx, :] = rt
                mask_reg[pidx, :] = rm

        return target_cls, target_reg, mask_reg

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Pass the gradient to their corresponding positions
        '''
        for i, r in enumerate(req):
            self.assign(in_grad[i], r, 0)


def _get_valid_labels(labels):
    #
    n_valid_label = 0
    for label in labels:
        if np.all(label == -1.0):
            break
        n_valid_label += 1
    return labels[:n_valid_label, :]


def _compute_iou(label, anchors_t, area_anchors_t):
    #
    iw = mx.nd.minimum(label[2], anchors_t[2]) - mx.nd.maximum(label[0], anchors_t[0])
    ih = mx.nd.minimum(label[3], anchors_t[3]) - mx.nd.maximum(label[1], anchors_t[1])
    I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
    U = (label[3] - label[1]) * (label[2] - label[0]) + area_anchors_t

    iou = I / mx.nd.maximum((U - I), 1e-08)
    return iou.asnumpy() # (num_anchors, )


def _compute_loc_target(gt_bb, bb):
    loc_target = np.tile(np.reshape(gt_bb, (1, -1)), (bb.shape[0], 1))
    loc_mask = np.ones_like(loc_target)
    return loc_target, loc_mask


def _adjust_ratio(bb, ratio):
    #
    ww = bb[2] - bb[0]
    hh = bb[3] - bb[1]
    cx = (bb[0] + bb[2]) / 2.0
    cy = (bb[1] + bb[3]) / 2.0

    ww *= np.sqrt(ratio)
    hh /= np.sqrt(ratio)

    res = bb.copy()
    res[0] = cx - ww * 0.5
    res[1] = cy - hh * 0.5
    res[2] = cx + ww * 0.5
    res[3] = cy + hh * 0.5
    return res


def _autofit_ratio(bb, max_ratio=3.0):
    #
    ww = bb[2] - bb[0]
    hh = bb[3] - bb[1]
    cx = (bb[0] + bb[2]) / 2.0
    cy = (bb[1] + bb[3]) / 2.0

    ratio = ww / hh
    if ratio > max_ratio:
        hh = ww / max_ratio
    elif ratio < 1.0 / max_ratio:
        ww = hh / max_ratio

    res = bb.copy()
    res[0] = cx - ww * 0.5
    res[1] = cy - hh * 0.5
    res[2] = cx + ww * 0.5
    res[3] = cy + hh * 0.5
    return res


@mx.operator.register("yolo_target")
class YoloTargetProp(mx.operator.CustomOpProp):
    def __init__(self, th_iou=0.5, th_iou_neg=0.4, th_iou_pass=0.25):
        #
        super(YoloTargetProp, self).__init__(need_top_grad=False)
        self.th_iou = float(th_iou)
        self.th_iou_neg = float(th_iou_neg)
        self.th_iou_pass = float(th_iou_pass)

    def list_arguments(self):
        return ['anchors', 'label', 'probs_cls']

    def list_outputs(self):
        return ['target_reg', 'mask_reg', 'target_cls', 'target_rpn']

    def infer_shape(self, in_shape):
        n_batch, n_class, n_sample = in_shape[2]

        target_reg_shape = (n_batch, n_sample, 4)
        mask_reg_shape = target_reg_shape
        target_cls_shape = (n_batch, 1, n_sample)

        out_shape = [target_reg_shape, mask_reg_shape, target_cls_shape, target_cls_shape]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return YoloTarget(self.th_iou, self.th_iou_neg, self.th_iou_pass)
