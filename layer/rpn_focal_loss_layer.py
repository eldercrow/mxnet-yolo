import mxnet as mx
import numpy as np
from ast import literal_eval


class RPNFocalLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, alpha, gamma, normalize):
        super(RPNFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalize = normalize

        self.eps = 1e-08

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        self.assign(out_data[0], req[0], in_data[1])

        # for debug
        p = mx.nd.pick(in_data[1], in_data[2], axis=1, keepdims=True)
        rp = mx.nd.pick(in_data[4], in_data[2] > 0, axis=1, keepdims=True)

        p_fg = p * rp
        p_bg = p + rp - p*rp
        fg_mask = in_data[2] > 0

        p = p_fg * fg_mask + p_bg * (1 - fg_mask)

        ce = -mx.nd.log(mx.nd.maximum(p, self.eps))

        ce *= mx.nd.power(1 - p, self.gamma)
        ce *= (in_data[2] > 0) * self.alpha + (in_data[2] == 0) * (1 - self.alpha)

        self.assign(out_data[1], req[1], ce)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Reweight loss according to focal loss.
        '''
        cls_target = mx.nd.reshape(in_data[2], (0, 1, -1))
        p0 = mx.nd.pick(in_data[1], cls_target, axis=1, keepdims=True)
        rp = mx.nd.pick(in_data[4], cls_target > 0, axis=1, keepdims=True)

        p_fg = p0 * rp
        p_bg = p0 + rp - p0*rp
        fg_mask = cls_target > 0

        p = p_fg * fg_mask + p_bg * (1 - fg_mask)

        ce = -mx.nd.log(mx.nd.maximum(p, self.eps))

        v = (p * self.gamma * ce) + 1 - p
        u = 1 - p if self.gamma == 2.0 else mx.nd.power(1 - p, self.gamma - 1.0)
        a = (cls_target > 0) * self.alpha + (cls_target == 0) * (1 - self.alpha)
        gf = v * u * a

        n_class = in_data[0].shape[1]
        label_mask = mx.nd.one_hot(mx.nd.reshape(cls_target, (0, -1)), n_class,
                on_value=1, off_value=0)
        label_mask = mx.nd.transpose(label_mask, (0, 2, 1))

        g = (in_data[1] - label_mask) * gf
        g *= (cls_target >= 0) # (n_batch, n_class, n_anchor)

        obj_mask = mx.nd.one_hot(mx.nd.reshape(fg_mask, (0, -1)), 2,
                on_value=1, off_value=0)
        obj_mask = mx.nd.transpose(obj_mask, (0, 2, 1))

        gr = (in_data[4] - obj_mask) * gf
        gr *= (cls_target >= 0)

        # care fg and bg
        rp_fg = rp * fg_mask
        rp_bg = rp * (1 - fg_mask)
        rp = rp_fg + (1 - rp_bg)
        g *= rp

        p0_fg = p0 * fg_mask
        p0_bg = p0 * (1 - fg_mask)
        p0 = p0_fg + (1 - p0_bg)
        gr *= p0

        if self.normalize:
            norm = mx.nd.sum(fg_mask).asscalar() + in_data[0].shape[0]
            g /= norm
            gr /= norm

        self.assign(in_grad[0], req[0], g)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], gr)
        self.assign(in_grad[4], req[4], 0)


@mx.operator.register("rpn_focal_loss")
class RPNFocalLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, alpha=0.25, gamma=2.0, normalize=True):
        #
        super(RPNFocalLossProp, self).__init__(need_top_grad=False)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.normalize = bool(literal_eval(str(normalize)))

    def list_arguments(self):
        return ['cls_pred', 'cls_prob', 'cls_target', 'rpn_pred', 'rpn_prob']

    def list_outputs(self):
        return ['cls_prob', 'cls_loss']

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0], in_shape[2]]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return RPNFocalLoss(self.alpha, self.gamma, self.normalize)
