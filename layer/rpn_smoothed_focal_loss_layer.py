import mxnet as mx
import numpy as np
import logging
from ast import literal_eval


class RPNSmoothedFocalLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, alpha, gamma, normalize):
        super(RPNSmoothedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # self.w_reg = w_reg
        self.normalize = normalize

        self.eps = np.finfo(np.float32).eps

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        self.assign(out_data[0], req[0], in_data[1])
        th_prob0 = in_data[5].asscalar()
        th_prob = th_prob0 / in_data[0].shape[1]

        # for debug
        p = mx.nd.pick(in_data[1], in_data[2], axis=1, keepdims=True)
        rp = mx.nd.pick(in_data[4], in_data[2] > 0, axis=1, keepdims=True)

        p_fg = p * rp
        p_bg = p + rp - p*rp
        fg_mask = in_data[2] > 0

        p = p_fg * fg_mask + p_bg * (1 - fg_mask)

        ce = -mx.nd.log(mx.nd.maximum(p, self.eps))
        sce = -p / th_prob - np.log(th_prob) + 1

        mask = p > th_prob
        sce = mask * ce + (1 - mask) * sce # smoothed cross entropy

        sce *= mx.nd.power(1 - p, self.gamma)
        sce *= (in_data[2] > 0) * self.alpha + (in_data[2] == 0) * (1 - self.alpha)
        sce += th_prob0 * th_prob0 # regularizer

        self.assign(out_data[1], req[1], sce)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Reweight loss according to focal loss.
        '''
        cls_target = in_data[2]
        th_prob0 = in_data[5].asscalar()
        th_prob = th_prob0 / in_data[0].shape[1]
        fg_mask = cls_target > 0

        p0 = mx.nd.pick(in_data[1], cls_target, axis=1, keepdims=True)
        rp = mx.nd.pick(in_data[4], fg_mask, axis=1, keepdims=True)

        p_fg = p0 * rp
        p_bg = p0 + rp - p0*rp

        p = p_fg * fg_mask + p_bg * (1 - fg_mask)

        ce = -mx.nd.log(mx.nd.maximum(p, self.eps))
        sce = -p / th_prob - np.log(th_prob) + 1

        mask = p > th_prob
        sce = mask * ce + (1 - mask) * sce # smoothed cross entropy

        thp = mx.nd.maximum(p, th_prob)
        u = 1 - p if self.gamma == 2.0 else mx.nd.power(1 - p, self.gamma - 1.0)
        v = p * self.gamma * sce + (p / thp) * (1 - p)
        a = fg_mask * self.alpha + (cls_target == 0) * (1 - self.alpha)
        gf = v * u * a

        n_class = in_data[0].shape[1]
        label_mask = mx.nd.one_hot(mx.nd.reshape(cls_target, (0, -1)), n_class,
                on_value=1, off_value=0)
        label_mask = mx.nd.transpose(label_mask, (0, 2, 1))

        g = (in_data[1] - label_mask) * gf
        g *= (cls_target >= 0)

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

        g_th = mx.nd.minimum(p, th_prob) / th_prob / th_prob - 1.0 / th_prob
        g_th /= in_data[0].shape[1]
        g_th *= mx.nd.power(1 - p, self.gamma)
        g_th = mx.nd.sum(g_th) + cls_target.size * th_prob0 * 2.0 #* self.w_reg

        if self.normalize:
            norm = mx.nd.sum(fg_mask).asscalar() + in_data[0].shape[0]
            g /= norm
            gr /= norm
            g_th /= norm
        if mx.nd.uniform(0, 1, (1,)).asscalar() < 0.001:
            logging.getLogger().info('{}: current th_prob for smoothed CE = {}'.format( \
                    type(self).__name__, th_prob))
            # logging.getLogger().info('Current th_prob for smoothed CE: {}'.format(th_prob))

        self.assign(in_grad[0], req[0], g)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], gr)
        self.assign(in_grad[4], req[4], 0)
        self.assign(in_grad[5], req[5], g_th)


@mx.operator.register("rpn_smoothed_focal_loss")
class RPNSmoothedFocalLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, alpha=0.25, gamma=2.0, normalize=False):
        #
        super(RPNSmoothedFocalLossProp, self).__init__(need_top_grad=False)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        # self.w_reg = float(w_reg)
        self.normalize = bool(literal_eval(str(normalize)))

    def list_arguments(self):
        return ['cls_pred', 'cls_prob', 'cls_target', 'rpn_pred', 'rpn_prob', 'th_prob']

    def list_outputs(self):
        return ['cls_prob', 'cls_loss']

    def infer_shape(self, in_shape):
        # in_shape[3] = (1,)
        out_shape = [in_shape[0], in_shape[2]]
        return in_shape, out_shape, []

    # def infer_type(self, in_type):
    #     dtype = in_type[0]
    #     import ipdb
    #     ipdb.set_trace()
    #     return [dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return RPNSmoothedFocalLoss(self.alpha, self.gamma, self.normalize)
