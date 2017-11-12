# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"
import mxnet as mx
import numpy as np

from ast import literal_eval as make_tuple


class ROITransform(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, variances):
        #
        super(ROITransform, self).__init__()
        self.variances = variances

    def forward(self, is_train, req, in_data, out_data, aux):
        #
        n_batch = in_data[0].shape[0]
        preds = in_data[0]
        anchors = mx.nd.reshape(in_data[1], (-1, 4))

        # if not np.isfinite(mx.nd.sum(preds).asscalar()):
        #     import ipdb
        #     ipdb.set_trace()

        at = mx.nd.transpose(anchors)

        P = mx.nd.zeros_like(preds)
        S = mx.nd.zeros_like(preds)

        for i in range(n_batch):
            rt = mx.nd.transpose(preds[i]) # (4, n_anchor)
            P[i], S[i] = self._transform_roi(rt, at)

        self.assign(out_data[0], req[0], P)
        self.assign(out_data[1], req[1], S)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #
        n_batch = in_data[0].shape[0]
        dL_dp = mx.nd.reshape(out_grad[0], (-1, 4))
        S = mx.nd.reshape(out_data[1], (-1, 4))

        gt = mx.nd.transpose(dL_dp)
        st = mx.nd.transpose(S)

        rt = mx.nd.zeros_like(gt)
        rt[0] = (gt[0] + gt[2]) * st[0]
        rt[1] = (gt[1] + gt[3]) * st[1]
        rt[2] = (gt[2] - gt[0]) * st[2]
        rt[3] = (gt[3] - gt[1]) * st[3]

        self.assign(in_grad[0], req[0], mx.nd.reshape(mx.nd.transpose(rt), in_grad[0].shape))
        self.assign(in_grad[1], req[1], 0)

    def _transform_roi(self, rt, at):
        #
        cx = (at[0] + at[2]) * 0.5
        cy = (at[1] + at[3]) * 0.5
        aw = at[2] - at[0]
        ah = at[3] - at[1]

        scaler = mx.nd.zeros_like(rt)
        scaler[0] = self.variances[0] * aw
        scaler[1] = self.variances[1] * ah
        # if mx.nd.max(rt[2]).asscalar() > 100:
        #     import ipdb
        #     ipdb.set_trace()
        # if mx.nd.max(rt[3]).asscalar() > 100:
        #     import ipdb
        #     ipdb.set_trace()
        scaler[2] = self.variances[2] * 0.5 * aw * mx.nd.exp(rt[2] * self.variances[2])
        scaler[3] = self.variances[3] * 0.5 * ah * mx.nd.exp(rt[3] * self.variances[3])

        cx += rt[0] * scaler[0]
        cy += rt[1] * scaler[1]
        w2 = scaler[2] / self.variances[2]
        h2 = scaler[3] / self.variances[3]

        rt[0] = cx - w2
        rt[1] = cy - h2
        rt[2] = cx + w2
        rt[3] = cy + h2
        return mx.nd.transpose(rt), mx.nd.transpose(scaler)


@mx.operator.register('roi_transform')
class ROITransformProp(mx.operator.CustomOpProp):
    def __init__(self, variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(ROITransformProp, self).__init__(need_top_grad=True)
        self.variances = make_tuple(str(variances))

    def list_arguments(self):
        return ['loc_deltas', 'anchors']

    def list_outputs(self):
        return ['loc_preds', 'scalers']

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0], in_shape[0]]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return ROITransform(self.variances)
