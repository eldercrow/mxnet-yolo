# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"
import mxnet as mx
import numpy as np


class MergeRPNCLS(mx.operator.CustomOp):
    '''
    '''
    def __init__(self):
        #
        super(MergeRPNCLS, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        #
        cls = in_data[0].copy()
        rpn = in_data[1]

        for i in range(cls.shape[0]):
            c = cls[i]
            r = rpn[i]

            c[0] = c[0] + r[0] - c[0]*r[0]
            c[1:] *= r[1:]
            cls[i] = c

        self.assign(out_data[0], req[0], cls)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('merge_rpn_cls')
class MergeRPNCLSProp(mx.operator.CustomOpProp):
    def __init__(self):
        #
        super(MergeRPNCLSProp, self).__init__()

    def list_arguments(self):
        return ['conv_cls', 'conv_rpn']

    def list_outputs(self):
        return ['merged_cls',]

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0],]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return MergeRPNCLS()
