import mxnet as mx
# from symbol.net_block_sep import *


def pool(data, name=None, kernel=(3, 3), pad=(1, 1), stride=(2, 2), pool_type='max'):
    return mx.sym.Pooling(data, name=name, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type)


def relu_conv_bn(data, prefix_name,
                 num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1,
                 wd_mult=1.0, no_bias=True, use_crelu=False,
                 use_global_stats=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'

    relu = mx.sym.Activation(data, act_type='relu')

    conv_w = mx.sym.var(name=conv_name+'_weight', lr_mult=1.0, wd_mult=wd_mult)
    conv_b = None if no_bias else mx.sym.var(name=conv_name+'_bias', lr_mult=2.0, wd_mult=0.0)
    conv = mx.sym.Convolution(relu, name=conv_name, weight=conv_w, bias=conv_b,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, num_group=num_group,
            no_bias=no_bias)
    if use_crelu:
        conv = mx.sym.concat(conv, -conv)

    bn = mx.sym.BatchNorm(conv, name=bn_name,
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04, momentum=0.99)
    return bn


def depthwise_conv(data, name, num_filter,
        kernel=(3, 3), pad=(1, 1), stride=(1, 1),
        use_global_stats=False):
    #
    bn_1x1 = relu_conv_bn(data, name+'1x1/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    bn_3x3 = relu_conv_bn(bn_1x1, name+'3x3/',
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, num_group=num_filter,
            wd_mult=0.01, use_global_stats=use_global_stats)
    return bn_3x3


def subpixel_upsample(data, ch, c, r, name=None):
    '''
    Transform input data shape of (n, ch*r*c, h, w) to (n, ch, h*r, c*w).

    ch: number of channels after upsample
    r: row scale factor
    c: column scale factor
    '''
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0))  # (n*ch*r*c, h, w)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, r * c, 0, 0))  # (n*ch, r*c, h, w)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1))  # (n*ch, w, h, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c))  # (n*ch, w, h*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3))  # (n*ch, h*r, w, c)
    X = mx.sym.reshape(data=X, name=name, shape=(-4, -1, ch, 0, -3))  # (n, ch, h*r, w*c)
    return X


def inception(data, name, f1, f3, f5, fm, do_pool, use_global_stats):
    #
    kernel = (3, 3)
    stride = (2, 2) if do_pool else (1, 1)

    # used in conv1 and convm
    pool1 = pool(data, name=name+'pool1/', kernel=(3, 3), pad=(1, 1)) if do_pool else data

    # conv1
    conv1 = relu_conv_bn(pool1, name+'conv1/',
            num_filter=f1, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    # conv3
    conv3 = depthwise_conv(data, name+'conv3/',
            num_filter=f3, kernel=(3, 3), pad=(1, 1), stride=stride,
            use_global_stats=use_global_stats)

    # conv5
    conv5 = depthwise_conv(data, name+'conv5/',
            num_filter=f3, kernel=(5, 5), pad=(2, 2), stride=stride,
            use_global_stats=use_global_stats)

    # convm_1 = pool1
    convm_1 = pool(pool1, kernel=(3, 3), pad=(1, 1), name=name+'convm_1')
    convm_2 = depthwise_conv(convm_1, name+'convm_2/',
            num_filter=fm*4, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    convm_3 = subpixel_upsample(convm_2, fm, 2, 2)

    concat = mx.sym.concat(conv1, conv3, conv5, convm_3)
    return concat
    # convc = relu_conv_bn(concat, name+'convc/',
    #         num_filter=f1+f3+f5+fm, kernel=(1, 1), pad=(0, 0),
    #         use_global_stats=use_global_stats)
    #
    # return convc


def proj_add(lhs, rhs, name, num_filter, do_pool, use_global_stats):
    #
    lhs = pool(lhs, kernel=(3, 3), pad=(1, 1)) if do_pool else lhs
    lhs = relu_conv_bn(lhs, name+'lhs/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    rhs = relu_conv_bn(rhs, name+'rhs/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    return mx.sym.broadcast_add(lhs, rhs, name=name+'add/')


def topdown_feature(data, updata, name, scale, nch_up, nf_proj, nf_all, use_global_stats):
    #
    # upsample, proj, concat, mix
    updata = mx.sym.UpSampling(updata, scale=scale, sample_type='bilinear',
            num_filter=nch_up, name=name+'upsample')
    updata = relu_conv_bn(updata, name+'proj/',
            num_filter=nf_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    data = mx.sym.concat(data, updata, name=name+'concat')
    data = depthwise_conv(data, name+'mix/',
            num_filter=nf_all, use_global_stats=use_global_stats)
    return data


def prepare_groups(data, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3 1
    f1 = [48, 96, 192]
    f3 = [48, 96, 192]
    f5 = [48, 96, 192]
    fm = [48, 96, 192]
    fa = [192, 384, 768]
    nu = [2, 4, 2]

    groups = []
    inci = data
    for i in range(3):
        inc0 = inci
        for j in range(nu[i]):
            inci = inception(inci, 'inc{}/{}/'.format(i+3, j+1),
                    f1=f1[i], f3=f3[i], f5=f5[i], fm=fm[i], do_pool=(j == 0),
                    use_global_stats=use_global_stats)
        inci = proj_add(inc0, inci, 'inc{}/'.format(i+3),
                num_filter=fa[i], do_pool=True, use_global_stats=use_global_stats)
        groups.append(inci)

    # top-down features
    # groups[0] = topdown_feature(groups[0], groups[2], 'up0/', scale=4,
    #         nch_up=768, nf_proj=192, nf_all=384, use_global_stats=use_global_stats)
    #
    # groups[2] = depthwise_conv(groups[2], 'inc5/dil/',
    #         num_filter=384, use_global_stats=use_global_stats)

    # for SSD
    # g = groups[2]
    #
    # g = depthwise_conv(g, 'g3/1/',
    #         num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
    #         use_global_stats=use_global_stats)
    # g = depthwise_conv(g, 'g3/2/',
    #         num_filter=512, use_global_stats=use_global_stats)
    # groups.append(g)
    #
    # g = depthwise_conv(g, 'g4/1/',
    #         num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
    #         use_global_stats=use_global_stats)
    # g = depthwise_conv(g, 'g4/2/',
    #         num_filter=512, use_global_stats=use_global_stats)
    # groups.append(g)
    #
    # g = depthwise_conv(g, 'g5/1/',
    #         num_filter=512, pad=(0, 0), use_global_stats=use_global_stats)
    # groups.append(g)

    return groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.sym.Convolution(data, name='1/conv',
            num_filter=16, kernel=(3, 3), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = mx.sym.BatchNorm(concat1, name='1/bn',
            use_global_stats=use_global_stats, fix_gamma=False, momentum=0.99)

    bn2 = depthwise_conv(bn1, '2_1/',
            num_filter=64, use_global_stats=use_global_stats)
    bn2 = depthwise_conv(bn2, '2_2/',
            num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)

    bn3 = depthwise_conv(bn2, '3_1/',
            num_filter=128, use_global_stats=use_global_stats)
    bn3 = depthwise_conv(bn3, '3_2/',
            num_filter=128, use_global_stats=use_global_stats)

    groups = prepare_groups(bn3, use_global_stats)
    conv5 = mx.sym.Activation(groups[1], act_type='relu', name='relu_group1')
    conv6 = mx.sym.Activation(groups[2], act_type='relu', name='relu_group2')
    return conv5, conv6
    #
    # # from the original classification network
    # pool6 = mx.sym.Pooling(groups[2], name='pool6', kernel=(2, 2), pool_type='max')
    # fc7 = mx.sym.Convolution(pool6, name='fc7',
    #         num_filter=num_classes, kernel=(3, 3), pad=(0, 0), no_bias=False)
    # flatten = mx.sym.Flatten(fc7, name='flatten')
    #
    # softmax = mx.sym.SoftmaxOutput(flatten, name='softmax')
    # return softmax
