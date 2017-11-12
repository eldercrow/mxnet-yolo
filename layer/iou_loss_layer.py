# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"
import mxnet as mx
import numpy as np

class IOULoss(mx.operator.CustomOp):
    '''
    Compute IOU loss between GT and predicted boxes.
    '''
    def __init__(self, eps):
        #
        super(IOULoss, self).__init__()
        self.eps = eps

    def forward(self, is_train, req, in_data, out_data, aux):
        #
        n_batch = in_data[0].shape[0]
        preds = in_data[0] #mx.nd.reshape(in_data[0], (n_batch, -1, 4)) # (n_batch * n_anchor, 4)
        labels = in_data[1] #mx.nd.reshape(in_data[1], (n_batch, -1, 4))
        masks = in_data[2] #mx.nd.reshape(in_data[2], (n_batch, -1, 4))

        loss_iou = mx.nd.zeros((preds.shape[0], preds.shape[1]), ctx=preds.context)
        ix = mx.nd.zeros_like(loss_iou)
        iy = mx.nd.zeros_like(loss_iou)
        U = mx.nd.zeros_like(loss_iou)

        for i in range(n_batch):
            pt = mx.nd.transpose(preds[i]) # (4, n_anchor)
            gt = mx.nd.transpose(labels[i]) # (4, n_anchor)

            # compute IOU
            iou, ix[i], iy[i], U[i] = self._compute_iou(pt, gt)
            # if mx.nd.max(iou).asscalar() > 1:
            #     import ipdb
            #     ipdb.set_trace()
            loss_iou[i] = -mx.nd.log(mx.nd.maximum(iou, self.eps))

        loss_iou *= mx.nd.max(masks, axis=2)

        self.assign(out_data[0], req[0], mx.nd.reshape(loss_iou, (n_batch, -1)))
        self.assign(out_data[1], req[1], mx.nd.reshape(ix, (n_batch, -1)))
        self.assign(out_data[2], req[2], mx.nd.reshape(iy, (n_batch, -1)))
        self.assign(out_data[3], req[3], mx.nd.reshape(U, (n_batch, -1)))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #
        n_batch = in_data[0].shape[0]
        preds = in_data[0] #mx.nd.reshape(in_data[0], (n_batch, -1, 4)) # (n_batch * n_anchor, 4)
        labels = in_data[1] #mx.nd.reshape(in_data[1], (n_batch, -1, 4))
        masks = in_data[2] #mx.nd.reshape(in_data[2], (n_batch, -1, 4))

        ix = out_data[1] #mx.nd.reshape(out_data[1], (n_batch, -1))
        iy = out_data[2] #mx.nd.reshape(out_data[2], (n_batch, -1))
        I = ix * iy
        U = out_data[3] #mx.nd.reshape(out_data[3], (n_batch, -1,))

        dL = mx.nd.zeros_like(preds)

        for i in range(n_batch):
            pt = mx.nd.transpose(preds[i])
            gt = mx.nd.transpose(labels[i])

            dp_dr = pt[3] - pt[1]
            dp_dl = -dp_dr
            dp_db = pt[2] - pt[0]
            dp_du = -dp_db

            di_dr = iy[i] * (gt[2] > pt[2])
            di_dl = -iy[i] * (pt[0] > gt[0])
            di_db = ix[i] * (gt[3] > pt[3])
            di_du = -ix[i] * (pt[1] > gt[1])

            iU = (I[i] > 0) * 1.0 / U[i]
            iUI = (I[i] > 0) * (U[i] + I[i])
            iUI /= mx.nd.maximum(U[i] * I[i], self.eps)
            dL_dr = iU * dp_dr - iUI * di_dr
            dL_dl = iU * dp_dl - iUI * di_dl
            dL_db = iU * dp_db - iUI * di_db
            dL_du = iU * dp_du - iUI * di_du

            dL[i] = mx.nd.transpose(mx.nd.stack(dL_dl, dL_du, dL_dr, dL_db))
            # if not np.isfinite(mx.nd.sum(dL[i]).asscalar()):
            #     import ipdb
            #     ipdb.set_trace()

        dL *= masks
        dL /= mx.nd.sum(masks) / 4 # normalization
        # dL = mx.nd.reshape(dL, (n_batch, -1))

        self.assign(in_grad[0], req[0], dL)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

    def _compute_iou(self, lhs, rhs):
        #
        ix = mx.nd.minimum(lhs[2], rhs[2]) - mx.nd.maximum(lhs[0], rhs[0])
        ix = mx.nd.maximum(0, ix)
        iy = mx.nd.minimum(lhs[3], rhs[3]) - mx.nd.maximum(lhs[1], rhs[1])
        iy = mx.nd.maximum(0, iy)

        I = ix * iy

        U = (lhs[2] - lhs[0]) * (lhs[3] - lhs[1]) + \
                (rhs[2] - rhs[0]) * (rhs[3] - rhs[1])
        U = mx.nd.maximum(U - I, self.eps)
        iou = I / U
        return iou, ix, iy, U


@mx.operator.register('iou_loss')
class IOULossProp(mx.operator.CustomOpProp):
    def __init__(self, eps=1e-08):
        #
        super(IOULossProp, self).__init__(need_top_grad=False)
        self.eps = float(eps)

    def list_arguments(self):
        return ['loc_preds', 'loc_labels', 'loc_masks']

    def list_outputs(self):
        return ['iou_loss', 'ix', 'iy', 'U']

    def infer_shape(self, in_shape):
        n_batch = in_shape[0][0]
        n_data = in_shape[0][1]
        out_shape = [(n_batch, n_data) for _ in range(4)]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return IOULoss(self.eps)
