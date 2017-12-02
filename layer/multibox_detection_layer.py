import os
import mxnet as mx
import numpy as np
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
from ast import literal_eval


class MultiBoxDetection(mx.operator.CustomOp):
    '''
    python implementation of MultiBoxDetection layer.
    '''

    def __init__(self, th_pos, th_nms, nms_topk, variances, has_rpn):
        #
        super(MultiBoxDetection, self).__init__()
        self.th_pos = th_pos
        self.th_nms = th_nms
        self.nms_topk = nms_topk
        self.variances = variances
        self.has_rpn = has_rpn

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        pick positives, transform bbs, apply nms
        '''
        n_batch, n_class, n_anchor = in_data[0].shape

        probs_cls = in_data[0].copy()
        preds_reg = mx.nd.reshape(in_data[1], (n_batch, -1, 4))  # (n_batch, n_anchors, 4)
        anchors = mx.nd.reshape(in_data[2], (-1, 4))  # (n_anchors, 4)
        # im_scale = in_data[3]
        if self.has_rpn:
            probs_rpn = in_data[3]

        od = mx.nd.zeros_like(out_data[0])

        for nn in range(n_batch):
            out_i = mx.nd.transpose(od[nn], (1, 0))
            # out_i[:] = 0
            pcls = probs_cls[nn]  # (n_classes, n_anchors)
            if self.has_rpn:
                prpn = probs_rpn[nn]
                pcls[0] *= prpn[0]
                pcls[1:] *= prpn[1:]
            preg = preds_reg[nn]  # (n_anchor, 4)
            out_i[0] = mx.nd.argmax(pcls, axis=0) - 1
            max_pcls = mx.nd.max(pcls, axis=0)
            out_i[1] = max_pcls * (out_i[0] >= 0) * (max_pcls > self.th_pos)

            # if n_class == 1:
            #     iidx = mx.nd.reshape(pcls > self.th_pos, (-1,))
            #     out_i[0] = iidx - 1
            #     out_i[1][:] = mx.nd.reshape(pcls, (-1,))
            # else:
            #     out_i[1] = mx.nd.max(pcls, axis=0)
            #     iidx = out_i[1] > self.th_pos
            #     out_i[0] = iidx * (mx.nd.argmax(pcls, axis=0) + 1) - 1
            # iidx = mx.nd.array(np.where(iidx.asnumpy())[0])
            out_i[2:] = mx.nd.transpose(preg)
            out_i = _nms(out_i, self.th_nms, self.nms_topk)
            od[nn] = mx.nd.transpose(out_i, (1, 0)) # (1, n_anchor, 6)

        self.assign(out_data[0], req[0], od)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i, r in enumerate(req):
            self.assign(in_grad[i], r, 0)
        # pass


def _nms(out_t, th_nms, nms_topk):
    ''' GPU nms '''
    # out_t: (6, n_anchor)
    n_detection = out_t.shape[1]
    nms_mask = out_t[0].asnumpy()

    sidx = mx.nd.argsort(out_t[1], is_ascend=0).asnumpy()
    out = mx.nd.transpose(out_t)
    area_out_t = (out_t[4] - out_t[2]) * (out_t[5] - out_t[3])

    max_k = int(mx.nd.sum(out_t[1] > 0).asscalar())
    k = 0
    for ii in sidx[:max_k]:
        i = int(ii)
        cid = nms_mask[i]
        if cid < 0:
            continue
        iw = mx.nd.minimum(out_t[4][i], out_t[4]) - \
                mx.nd.maximum(out_t[2][i], out_t[2])
        ih = mx.nd.minimum(out_t[5][i], out_t[5]) - \
                mx.nd.maximum(out_t[3][i], out_t[3])
        I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
        iou_mask = (I / mx.nd.maximum(area_out_t + area_out_t[i] - I, 1e-06)) > th_nms
        cls_mask = out_t[0] == cid
        nidx = np.where((iou_mask * cls_mask).asnumpy())[0]

        # watch out the execution order!
        nms_mask[nidx] = -1
        nms_mask[i] = cid

        k += 1
        if k >= nms_topk:
            break
    out_t[0] = mx.nd.array(nms_mask)
    return out_t


def _transform_roi(reg, anc, variances):
    #
    # if iidx.size == 0:
    #     return reg
    # reg_t = mx.nd.transpose(mx.nd.take(reg, iidx))
    # anc_t = mx.nd.transpose(mx.nd.take(anc, iidx))
    reg_t = mx.nd.transpose(reg)
    anc_t = mx.nd.transpose(anc)

    for i in range(4):
        reg_t[i] *= variances[i]

    cx = (anc_t[0] + anc_t[2]) * 0.5
    cy = (anc_t[1] + anc_t[3]) * 0.5

    aw = anc_t[2] - anc_t[0]
    ah = anc_t[3] - anc_t[1]
    cx += reg_t[0] * aw
    cy += reg_t[1] * ah
    w = mx.nd.exp(reg_t[2]) * aw * 0.5
    h = mx.nd.exp(reg_t[3]) * ah * 0.5
    reg_t[0] = cx - w
    reg_t[1] = cy - h
    reg_t[2] = cx + w
    reg_t[3] = cy + h

    # reg_tt = mx.nd.transpose(reg_t)
    # for i, j in enumerate(iidx.asnumpy().astype(int)):
    #     reg[j] = reg_tt[i]
    reg = mx.nd.transpose(reg_t)
    return reg


@mx.operator.register("multibox_detection")
class MultiBoxDetectionProp(mx.operator.CustomOpProp):
    def __init__(self,
                 th_pos=0.1,
                 th_nms=0.35,
                 nms_topk=400,
                 variances=(0.1, 0.1, 0.2, 0.2),
                 has_rpn=False):
        #
        super(MultiBoxDetectionProp, self).__init__(need_top_grad=False)
        self.th_pos = float(th_pos)
        self.th_nms = float(th_nms)
        self.nms_topk = int(nms_topk)
        if isinstance(variances, str):
            variances = literal_eval(variances)
        self.variances = np.array(variances)
        self.has_rpn = literal_eval(str(has_rpn))

    def list_arguments(self):
        if not self.has_rpn:
            return ['probs_cls', 'preds_reg', 'anchors']
        else:
            return ['probs_cls', 'preds_reg', 'anchors', 'probs_rpn']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        n_batch, _, n_anchor = in_shape[0]
        out_shape = [(n_batch, n_anchor, 6)]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxDetection(self.th_pos, self.th_nms, self.nms_topk, self.variances, self.has_rpn)
