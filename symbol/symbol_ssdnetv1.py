import mxnet as mx


def conv_bn(data, name, \
        num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1, wd_mult=1.0, \
        crelu=False, use_global_stats=False):
    #
    conv_name = name + 'conv'
    bn_name = name + 'bn'
    conv_w = mx.sym.var(name=conv_name+'_weight', lr_mult=1.0, wd_mult=wd_mult)

    conv = mx.sym.Convolution(data, name=conv_name, weight=conv_w,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride,
            num_group=num_group, no_bias=True)
    if crelu:
        conv = mx.sym.concat(conv, -conv)
    bn = mx.sym.BatchNorm(conv, name=bn_name,
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04)
    return bn


def conv_bn_relu(data, name, \
        num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1, wd_mult=1.0, \
        crelu=False, use_global_stats=False):
    #
    bn = conv_bn(data, name,
            num_filter, kernel, pad, stride, num_group, wd_mult,
            crelu, use_global_stats)
    relu = mx.sym.Activation(bn, name=name+'relu', act_type='relu')
    return relu


def depthwise_unit(data, name, nf_dw, nf_sep,
        kernel=(3, 3), pad=(1, 1), stride=(1, 1),
        no_exp=False, use_global_stats=False):
    #
    if no_exp:
        conv_exp = data
    else:
        conv_exp = conv_bn_relu(data, name=name+'exp/',
                num_filter=nf_dw, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)

    conv_dw = conv_bn_relu(conv_exp, name=name+'dw/', \
            num_filter=nf_dw, kernel=kernel, pad=pad, stride=stride,
            num_group=nf_dw, wd_mult=0.01, \
            use_global_stats=use_global_stats)

    conv_sep = conv_bn(conv_dw, name=name+'sep/', \
            num_filter=nf_sep, kernel=(1, 1), pad=(0, 0), num_group=1, \
            use_global_stats=use_global_stats)
    return conv_sep


def inception(data, name, nf0, exp_factor, nfc, do_pool=False, use_global_stats=False):
    #
    units = []

    nf1 = nf0

    if do_pool:
        u0 = mx.sym.Pooling(data, name=name+'mp/',
                kernel=(4, 4), pad=(1, 1), stride=(2, 2), pool_type='max')
        u = depthwise_unit(data, name+'u0/',
                nf_dw=nf0*exp_factor, nf_sep=nf1, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
                use_global_stats=use_global_stats)
        u = u + u0
    else:
        u = depthwise_unit(data, name+'u0/',
                nf_dw=nf0*exp_factor, nf_sep=nf1,
                use_global_stats=use_global_stats)

    units.append(u)

    nf1 = nf1 // 2

    u = depthwise_unit(u, name+'u1/',
            nf_dw=nf0*exp_factor, nf_sep=nf1,
            use_global_stats=use_global_stats)
    units.append(u)

    nf0 = nf0 // 2

    u = depthwise_unit(u, name+'u2/',
            nf_dw=nf0*exp_factor, nf_sep=nf1,
            use_global_stats=use_global_stats)
    units.append(u)

    concat = mx.sym.concat(*units)
    concat = conv_bn(concat, name+'concat/',
            num_filter=nfc, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    return concat


def conv_group(data, name, n_unit, nf_dw, exp_factor, nf_sep, do_pool, use_global_stats):
    #
    nf_exp = nf_dw * exp_factor

    u0 = inception(data, name+'inc0/',
            nf0=nf_dw, exp_factor=exp_factor, nfc=nf_sep, do_pool=do_pool,
            use_global_stats=use_global_stats)

    for i in range(1, n_unit):
        #
        u = inception(u0, name+'inc{}/'.format(i),
                nf0=nf_dw, exp_factor=exp_factor, nfc=nf_sep,
                use_global_stats=use_global_stats)
        u0 = u + u0
    return u0


def get_symbol(num_classes=1000, **kwargs):
    #
    if 'use_global_stats' not in kwargs:
        use_global_stats = False
    else:
        use_global_stats = kwargs['use_global_stats']

    data = mx.sym.var('data')

    conv1 = conv_bn_relu(data, '1/',
            num_filter=32, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            crelu=True, use_global_stats=use_global_stats)

    pool2 = mx.sym.Pooling(conv1, name='2/pool', kernel=(2, 2), stride=(2, 2), pool_type='max')

    conv2 = depthwise_unit(pool2, '2/',
            nf_dw=64, nf_sep=24, no_exp=True,
            use_global_stats=use_global_stats)

    # (n_unit, nf_dw, exp_factor, nf_sep, do_pool, crelu)
    params = [ \
            (2, 24, 6, 48, True),
            (3, 48, 6, 96, True),
            (3, 96, 6, 192, True),
            ]

    g = conv2
    for i, p in enumerate(params, 3):
        n_unit, nf_dw, ef, nf_sep, do_pool = p
        g = conv_group(g, '{}/'.format(i),
                n_unit, nf_dw, ef, nf_sep, do_pool, use_global_stats)

    convf = conv_bn_relu(g, 'f/',
            num_filter=192*6, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    poolf = mx.sym.Pooling(convf, name='poolf', kernel=(1, 1),
            global_pool=True, pool_type='avg', pooling_convention='full')

    fc7 = mx.sym.Convolution(poolf, name='fc',
            num_filter=num_classes, kernel=(1, 1), pad=(0, 0), no_bias=False)
    flatten = mx.sym.flatten(fc7, name='flatten')

    softmax = mx.sym.SoftmaxOutput(flatten, name='softmax')
    return softmax


