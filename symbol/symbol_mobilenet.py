import mxnet as mx


def batchnorm(data, name, use_global_stats=False):
    return mx.sym.BatchNorm(data, name=name, use_global_stats=use_global_stats,
            fix_gamma=False, eps=1e-04, momentum=0.99)


def conv_bn(data, name, \
        num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1, wd_mult=1.0, \
        use_global_stats=False):
    #
    conv_name='conv'+name
    conv_w = mx.sym.var(name=conv_name+'_weight', lr_mult=1.0, wd_mult=wd_mult)

    conv = mx.sym.Convolution(data, name=conv_name, weight=conv_w, num_filter=num_filter, \
            kernel=kernel, pad=pad, stride=stride, num_group=num_group, no_bias=True)
    bn = batchnorm(conv, name=conv_name+'_bn', use_global_stats=use_global_stats)
    return bn


def conv_bn_relu(data, name, \
        num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1, wd_mult=1.0, \
        use_global_stats=False):
    #
    conv_name='conv'+name
    conv_w = mx.sym.var(name=conv_name+'_weight', lr_mult=1.0, wd_mult=wd_mult)

    conv = mx.sym.Convolution(data, name=conv_name, weight=conv_w, num_filter=num_filter, \
            kernel=kernel, pad=pad, stride=stride, num_group=num_group, no_bias=True)
    bn = batchnorm(conv, name=conv_name+'_bn', use_global_stats=use_global_stats)
    relu = mx.sym.Activation(bn, name='relu'+name, act_type='relu')
    return relu


def depthwise_unit(data, name, nf_dw, nf_sep,
        kernel=(3, 3), pad=(1, 1), stride=(1, 1),
        no_act=False, use_global_stats=False):
    #
    if not no_act:
        conv_dw = conv_bn_relu(data, name=name+'_dw', \
                num_filter=nf_dw, kernel=kernel, pad=pad, stride=stride, num_group=nf_dw, \
                wd_mult=0.0, use_global_stats=use_global_stats)
    else:
        conv_dw = conv_bn(data, name=name+'_dw', \
                num_filter=nf_dw, kernel=kernel, pad=pad, stride=stride, num_group=nf_dw, \
                wd_mult=0.0, use_global_stats=use_global_stats)

    if nf_sep == 0:
        return conv_dw
    conv_sep = conv_bn_relu(conv_dw, name=name+'_sep', \
            num_filter=nf_sep, kernel=(1, 1), pad=(0, 0), \
            use_global_stats=use_global_stats)
    return conv_sep


def subpixel_downsample(data, ch, c, r, name=''):
    '''
    '''
    if r == 1 and c == 1:
        return data
    X = mx.sym.transpose(data, axes=(0, 2, 3, 1)) # (n, w, h, ch)
    X = mx.sym.reshape(X, shape=(0, 0, -1, ch*c), name=name+'_rch') # (n, w, h*r, ch/r)
    X = mx.sym.transpose(X, axes=(0, 2, 1, 3)) # (n, h*r, w, ch/r)
    X = mx.sym.reshape(X, shape=(0, 0, -1, ch*c*r), name=name+'_rcch') # (n, h*r, w*c, ch/r/c)
    X = mx.sym.transpose(X, axes=(0, 3, 2, 1)) # (n, ch/r/c, h*r, w*c)
    return X


def get_symbol(num_classes=1000, use_global_stats=False):
    #
    data = mx.sym.var(name='data')
    label = mx.sym.var(name='label')

    conv1 = conv_bn_relu(data, '1',
            num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)

    nf_dw_all = [(32, 64), (128, 128), (256, 256), (512, 512, 512, 512, 512, 512)]
    nf_sep_all = [(64, 128), (128, 256), (256, 512), (512, 512, 512, 512, 512, 1024)]
    stride_all = [(1, 2), (1, 2), (1, 2), (1, 1, 1, 1, 1, 2)]

    convi = conv1
    groups = []
    for i, (nf_dw_i, nf_sep_i, stride_i) in enumerate(zip(nf_dw_all, nf_sep_all, stride_all), 2):
        for j, (nf_dw, nf_sep, stride) in enumerate(zip(nf_dw_i, nf_sep_i, stride_i), 1):
            name = '{}_{}'.format(i, j)
            ss = (stride, stride)
            convi = depthwise_unit(convi, name,
                    nf_dw=nf_dw, nf_sep=nf_sep, stride=ss, use_global_stats=use_global_stats)
            if i > 3 and j == len(nf_dw_i)-1:
                groups.append(convi)

    conv6 = depthwise_unit(convi, '6', nf_dw=1024, nf_sep=1024, use_global_stats=use_global_stats)

    # from the original classification network
    pool6 = mx.sym.Pooling(conv6, name='pool6', kernel=(1, 1), global_pool=True, pool_type='avg')
    fc7 = mx.sym.Convolution(pool6, name='fc7',
            num_filter=num_classes, kernel=(1, 1), pad=(0, 0), no_bias=False)
    flatten = mx.sym.Flatten(fc7, name='flatten')

    softmax = mx.sym.SoftmaxOutput(flatten, name='softmax')
    return softmax
