import mxnet as mx
import numpy as np
from collections import Iterable
from ast import literal_eval as make_tuple


class AnchorBox(mx.operator.CustomOp):
    '''
    Python alternative of AnchorBox class.
    Will handle anchor box layer in a different way.
    Also I will handle sizes and ratios in a different - like rcnn - way.
    '''
    def __init__(self, anc_shapes, clip):
        super(AnchorBox, self).__init__()
        self.anc_shapes = anc_shapes
        self.clip = clip
        self.anchor_data = None

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data:
            a conv layer that we will infer the size of outputs.
        out_data:
            anchors (1 num_anchor*4 h w)
        '''
        if self.anchor_data is not None:
            self.assign(out_data[0], req[0], self.anchor_data)
            return

        anchors_all = np.empty((0, 4), dtype=np.float32)

        h = in_data[0].shape[2]
        w = in_data[0].shape[3]

        apc = self.anc_shapes.shape[0]

        stride = (1.0 / w, 1.0 / h)

        # compute center positions
        x = (np.arange(w) + 0.5) * stride[0]
        y = (np.arange(h) + 0.5) * stride[1]
        xv, yv = np.meshgrid(x, y)

        # compute heights and widths
        wh = np.zeros((apc, 4))
        for k, anc in enumerate(self.anc_shapes):
            wh[k, 0] = -(anc[0] * stride[0] * 0.5)
            wh[k, 1] = -(anc[1] * stride[1] * 0.5)
            wh[k, 2] =  (anc[0] * stride[0] * 0.5)
            wh[k, 3] =  (anc[1] * stride[1] * 0.5)
            # wh[k, 0] = -(anc[0] * 0.5)
            # wh[k, 1] = -(anc[1] * 0.5)
            # wh[k, 2] =  (anc[0] * 0.5)
            # wh[k, 3] =  (anc[1] * 0.5)

        # build anchors
        anchors = np.zeros((h, w, apc, 4), dtype=np.float32)
        for i in range(apc):
            anchors[:, :, i, 0] = xv + wh[i, 0]
            anchors[:, :, i, 1] = yv + wh[i, 1]
            anchors[:, :, i, 2] = xv + wh[i, 2]
            anchors[:, :, i, 3] = yv + wh[i, 3]

        anchors = np.reshape(anchors, (-1, 4))
        anchors_all = np.vstack((anchors_all, anchors))

        if self.clip > 0:
            anchors_all = np.minimum(np.maximum(anchors_all, 0.0), 1.0)
        anchors_all = np.reshape(anchors_all, (1, -1, 4))

        self.anchor_data = mx.nd.array(anchors_all, ctx=in_data[0].context)
        self.assign(out_data[0], req[0], self.anchor_data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register("anchor_box")
class AnchorBoxProp(mx.operator.CustomOpProp):
    def __init__(self, shapes, clip=False):
        super(AnchorBoxProp, self).__init__(need_top_grad=False)
        anc_shapes = make_tuple(shapes)
        assert len(anc_shapes) % 2 == 0
        self.anc_shapes = np.array(anc_shapes).reshape(-1, 2)
        self.clip = int(clip)

    def list_arguments(self):
        return ['ref_conv',]

    def list_outputs(self):
        return ['anchors',]

    def infer_shape(self, in_shape):
        n_anc_shape = self.anc_shapes.shape[0]
        n_anchor = n_anc_shape * in_shape[0][2] * in_shape[0][3]
        return in_shape, [(1, n_anchor, 4),], []

    def create_operator(self, ctx, shapes, dtypes):
        return AnchorBox(self.anc_shapes, self.clip)
