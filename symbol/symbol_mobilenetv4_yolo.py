import mxnet as mx
from symbol_mobilenetv4 import get_symbol as get_mobilenet
from symbol_mobilenet import depthwise_unit, subpixel_downsample

def get_symbol(num_classes, use_global_stats):
    #
    conv5, conv6 = get_mobilenet(num_classes=num_classes, use_global_stats=use_global_stats)
    #
    # conv5_5 = bone.get_internals()['relu5_5_sep_output']
    # conv6 = bone.get_internals()['relu6_sep_output']

    # extra layers
    conv7_1 = depthwise_unit(conv6, '7_1',
            nf_dw=768, nf_sep=768, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    conv7_2 = depthwise_unit(conv7_1, '7_2',
            nf_dw=768, nf_sep=768, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)

    # re-organze conv5_5 and concat conv7_2
    conv5_1 = subpixel_downsample(conv5, 384, 2, 2)
    concat = mx.sym.concat(conv5_1, conv7_2)
    # concat = conv7_2
    conv8_1 = depthwise_unit(concat, '8_1',
            nf_dw=768*3, nf_sep=2048, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)

    return conv8_1
