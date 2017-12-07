import mxnet as mx
import numpy as np
from config.config import cfg
# import cv2


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=np.finfo(np.float32).eps):
        self.eps = eps

        super(MultiBoxMetric, self).__init__('MultiBox')
        self.num = 3
        self.name = ['CrossEntropy', 'NegLogIOU', 'RPNEntropy']

        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # # get generated multi label from network
        # cls_prob = mx.nd.pick(preds[0], preds[2], axis=1, keepdims=True).asnumpy()
        # # cls_prob = np.maximum(cls_prob, self.eps)
        # cls_label = preds[2].asnumpy()
        # loss = -np.log(np.maximum(cls_prob, self.eps))
        #
        # # focal loss reweighting
        # gamma = float(cfg.train['focal_loss_gamma'])
        # alpha = float(cfg.train['focal_loss_alpha'])
        #
        # # smoothed softmax
        # if cfg.train['use_smooth_ce']:
        #     w_reg = float(cfg.train['smooth_ce_lambda'])
        #     th_prob = preds[-1].asnumpy()[0] #float(cfg.train['smooth_ce_th'])
        #     loss1 = -cls_prob / th_prob - np.log(th_prob) + 1
        #     idx = cls_prob < th_prob
        #     loss[idx] = loss1[idx]
        #     loss += th_prob * th_prob * w_reg * preds[0].shape[1]
        #
        # loss *= np.power(1 - cls_prob, gamma)
        # loss[cls_label > 0] *= alpha
        # loss[cls_label ==0] *= 1 - alpha
        # loss *= (cls_label >= 0)

        loss = preds[0].asnumpy()
        cls_label = preds[2].asnumpy()
        loss *= (cls_label >= 0)

        self.sum_metric[0] += loss.sum()
        self.num_inst[0] += np.sum(cls_label > 0) + preds[0].shape[0]

        # -log IOU
        loc_loss = preds[1].asnumpy()
        if np.sum(loc_loss) < 0:
            import pdb
            pdb.set_trace()
        loc_label = preds[3].asnumpy()
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += np.sum(loc_label) / 4

        loss_rpn = preds[5].asnumpy()
        loss_rpn *= (cls_label >= 0)

        self.sum_metric[2] += loss_rpn.sum()
        self.num_inst[2] += np.sum(cls_label > 0) + preds[0].shape[0]

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
