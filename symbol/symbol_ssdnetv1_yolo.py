"""
Reference:
Redmon, Joseph, and Ali Farhadi. "YOLO9000: Better, Faster, Stronger."
"https://arxiv.org/pdf/1612.08242.pdf"
"""
import mxnet as mx
from symbol_ssdnetv1 import get_symbol as get_net
from symbol_ssdnetv1 import conv_bn_relu, depthwise_unit


def subpixel_upsample(data, ch, c, r, name=None):
    '''
    Transform input data shape of (n, ch, h, w) to (n, ch/c/r, h*r, c*w).

    ch: number of channels after upsample
    r: row scale factor
    c: column scale factor
    '''
    if r == 1 and c == 1:
        return data
    X = mx.sym.transpose(data, axes=(0, 3, 2, 1)) # (n, w, h, ch)
    X = mx.sym.reshape(X, shape=(0, 0, -1, ch/r)) # (n, w, h*r, ch/r)
    X = mx.sym.transpose(X, axes=(0, 2, 1, 3)) # (n, h*r, w, ch/r)
    X = mx.sym.reshape(X, shape=(0, 0, -1, ch/r/c)) # (n, h*r, w*c, ch/r/c)
    X = mx.sym.transpose(X, axes=(0, 3, 1, 2)) # (n, ch/r/c, h*r, w*c)
    return X


def get_symbol(num_classes, use_global_stats):
    #
    bone = get_net(num_classes=num_classes, use_global_stats=use_global_stats)

    g3 = bone.get_internals()['3/inc1/concat/bn_output']
    g4 = bone.get_internals()['4/inc2/concat/bn_output']
    g5 = bone.get_internals()['5/inc2/concat/bn_output']

    h0 = depthwise_unit(g3, 'h0/',
            nf_dw=48*6, nf_sep=48, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    h2 = depthwise_unit(g5, 'h2/',
            nf_dw=192*6, nf_sep=192, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    h2 = subpixel_upsample(h2, 192, 2, 2)

    hyperf = mx.sym.concat(h0, g4, h2, name='hyperf')

    # rpn and loc
    hyperf_rpn = depthwise_unit(hyperf, 'hyperf/rpn/',
            nf_dw=96*6, nf_sep=32,
            use_global_stats=use_global_stats)
    body_rpn = conv_bn_relu(hyperf_rpn, 'body_rpn/',
            num_filter=32*6, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    body_loc = conv_bn_relu(hyperf_rpn, 'body_loc/',
            num_filter=32*6, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    hyperf_rcnn = depthwise_unit(hyperf, 'hyperf/rcnn/fc1',
            nf_dw=192*6, nf_sep=192,
            use_global_stats=use_global_stats)
    hyperf_rcnn = depthwise_unit(hyperf_rcnn, 'hyperf/rcnn/fc2',
            nf_dw=192*6, nf_sep=192,
            use_global_stats=use_global_stats)

    body_cls = conv_bn_relu(hyperf_rcnn, 'body_cls/',
            num_filter=192*6, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    return body_rpn, body_loc, body_cls
