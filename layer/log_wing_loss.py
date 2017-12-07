# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"
import mxnet as mx
import numpy as np

class LogWingLoss(mx.operator.CustomOp):
    '''
    Compute IOU loss between GT and predicted boxes.
    '''
    def __init__(self, scalar):
        #
        super(LogWingLoss, self).__init__()
        assert scalar == 1.0
        self.scalar = scalar

    def forward(self, is_train, req, in_data, out_data, aux):
        #
        diffs = mx.nd.abs(in_data[0])
        self.mask = diffs < 1.0

        loss1 = 0.5 * diffs * diffs
        loss2 = mx.nd.log(mx.nd.maximum(diffs, 1.0)) + 0.5

        loss = loss1 * self.mask + loss2 * (1 - self.mask)
        self.assign(out_data[0], req[0], loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #
        diffs = in_data[0]

        # import pdb
        # pdb.set_trace()

        grad1 = diffs
        grad2 = 1.0 / mx.nd.maximum(mx.nd.abs(diffs), 1.0) * mx.nd.sign(diffs)

        grad = grad1 * self.mask + grad2 * (1 - self.mask)
        self.assign(in_grad[0], req[0], out_grad[0] * grad)


@mx.operator.register('log_wing_loss')
class LogWingLossProp(mx.operator.CustomOpProp):
    def __init__(self, scalar=1.0):
        #
        super(LogWingLossProp, self).__init__(need_top_grad=True)
        # TODO: relax this
        assert scalar == 1.0
        self.scalar = float(scalar)

    def list_arguments(self):
        return ['diff',]

    def list_outputs(self):
        return ['loss',]

    def infer_shape(self, in_shape):
        return in_shape, in_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return LogWingLoss(self.scalar)
