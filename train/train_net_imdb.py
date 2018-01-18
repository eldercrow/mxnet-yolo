import tools.find_mxnet
import mxnet as mx
import numpy as np
import logging
import sys
import os
import importlib
import re
import random
from dataset.iterator import DetIter
from train.metric import MultiBoxMetric
from evaluate.eval_metric import MApMetric, VOC07MApMetric
from config.config import cfg
from train.lr_scheduler import BurnInMultiFactorScheduler
from symbol.symbol_builder import get_symbol_train
from train.gnadam import GNadam
from tools.load_checkpoint import load_checkpoint
from tools.rand_sampler import RandScaler


def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    """
    Compute learning rate and refactor scheduler

    Parameters:
    ---------
    learning_rate : float
        original learning rate
    lr_refactor_step : comma separated str
        epochs to change learning rate
    lr_refactor_ratio : float
        lr *= ratio at certain steps
    num_example : int
        number of training images, used to estimate the iterations given epochs
    batch_size : int
        training batch size
    begin_epoch : int
        starting epoch

    Returns:
    ---------
    (learning_rate, mx.lr_scheduler) as tuple
    """
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        if not steps:
            return (lr, None)
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        # lr_scheduler = BurnInMultiFactorScheduler(burn_in=1000, step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)


def set_mod_params(mod, args, auxs, logger):
    mod.init_params(initializer=mx.init.Xavier())
    args0, auxs0 = mod.get_params()
    arg_params = args0.copy()
    aux_params = auxs0.copy()

    # for k, v in sorted(arg_params.items()):
    #     print k, v.shape

    if args is not None:
        for k in args0:
            if k in args and args0[k].shape == args[k].shape:
                arg_params[k] = args[k]
            else:
                logger.info('param {} is inited from scratch.'.format(k))
    else:
        logger.info('from scratch training mode, args are inited by random.')

    if auxs is not None:
        for k in auxs0:
            if k in auxs and auxs0[k].shape == auxs[k].shape:
                aux_params[k] = auxs[k]
            else:
                logger.info('param {} is inited from scratch.'.format(k))
    else:
        logger.info('from scratch training mode, auxs are inited by random.')

    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    return mod

def train_net(net, imdb,
              batch_size, data_shape, mean_pixels,
              resume, finetune, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent,
              optimizer_name='gnadam', learning_rate=5e-03, momentum=0.9, weight_decay=1e-04,
              lr_refactor_step=(80,120), lr_refactor_ratio=0.1,
              val_imdb=None,
              freeze_layer_pattern='',
              min_obj_size=12.0, use_difficult=False,
              random_aspect_exp=2.0, random_aspect_epoch=5, img_stride=32,
              nms_thresh=0.45, force_nms=False, ovp_thresh=0.5,
              voc07_metric=False, nms_topk=400,
              iter_monitor=0, monitor_pattern=".*", log_file=None):
    """
    Wrapper for training phase.

    Parameters:
    ----------
    net : str
        symbol name for the network structure
    imdb : str
        imdb for training
    batch_size : int
        training batch-size
    data_shape : int or tuple
        width/height as integer or (3, height, width) tuple
    mean_pixels : tuple of floats
        mean pixel values for red, green and blue
    resume : int
        resume from previous checkpoint if > 0
    finetune : int
        fine-tune from previous checkpoint if > 0
    pretrained : str
        prefix of pretrained model, including path
    epoch : int
        load epoch of either resume/finetune/pretrained model
    prefix : str
        prefix for saving checkpoints
    ctx : [mx.cpu()] or [mx.gpu(x)]
        list of mxnet contexts
    begin_epoch : int
        starting epoch for training, should be 0 if not otherwise specified
    end_epoch : int
        end epoch of training
    frequent : int
        frequency to print out training status
    learning_rate : float
        training learning rate
    momentum : float
        trainig momentum
    weight_decay : float
        training weight decay param
    lr_refactor_ratio : float
        multiplier for reducing learning rate
    lr_refactor_step : comma separated integers
        at which epoch to rescale learning rate, e.g. '30, 60, 90'
    freeze_layer_pattern : str
        regex pattern for layers need to be fixed
    random_aspect_exp : float
        random aspect ratio exponent
    random_aspect epoch : int
        number of epoch before next random aspect
    nms_thresh : float
        non-maximum suppression threshold for validation
    force_nms : boolean
        suppress overlaped objects from different classes
    val_imdb : str
        imdb for validation
    iter_monitor : int
        monitor internal stats in networks if > 0, specified by monitor_pattern
    monitor_pattern : str
        regex pattern for monitoring network stats
    log_file : str
        log to file if enabled
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # check args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    prefix += '_' + net.strip('_yolo') + '_' + str(data_shape[1])

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"

    # label padding
    if val_imdb:
        max_objects = max(imdb.max_objects, val_imdb.max_objects)
        imdb.pad_labels(max_objects)
        val_imdb.pad_labels(max_objects)

    class_names = imdb.classes
    num_example = imdb.num_images
    num_classes = len(class_names)

    # load symbol
    # sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    net = get_symbol_train('symbol_'+net, num_classes, nms_thresh, force_nms, nms_topk)

    # define layers with fixed weight/bias
    if freeze_layer_pattern.strip():
        re_prog = re.compile(freeze_layer_pattern)
        fixed_param_names = [name for name in net.list_arguments() if re_prog.match(name)]
    else:
        fixed_param_names = None

    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    allow_missing = True
    if resume > 0:
        logger.info("Resume training with {} from epoch {}"
            .format(ctx_str, resume))
        _, args, auxs = mx.model.load_checkpoint(prefix, resume)
        begin_epoch = resume
        allow_missing = False
    elif pretrained:
        try:
            logger.info("Start training with {} from pretrained model {}"
                .format(ctx_str, pretrained))
            args, auxs = load_checkpoint(pretrained, epoch)
        except:
            logger.info("Failed to load the pretrained model. Start from scratch.")
            args = None
            auxs = None
            fixed_param_names = None
    else:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None

    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # init training module
    mod = mx.mod.Module(net, label_names=('yolo_output_label',), logger=logger, context=ctx,
                        fixed_param_names=fixed_param_names)
    #
    # for debug, DO NOT DELETE!
    # # print_output_shapes(symbol, train_iter.provide_data[0][1], train_iter.provide_label[0][1])
    # syms = mod.symbol.get_internals()
    # _, out_shapes, _ = syms.infer_shape_partial( \
    #         data=train_iter.provide_data[0][1], label=train_iter.provide_label[0][1])
    # for oname, oshape in zip(syms.list_outputs(), out_shapes):
    #     if 'output' in oname:
    #         print oname, oshape
    #

    # fit parameters
    batch_end_callback = mx.callback.Speedometer(batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None
    optimizer_params={'learning_rate': learning_rate,
                      'wd': weight_decay,
                      'clip_gradient': 4.0,
                      'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0 }
    if optimizer_name == 'sgd':
        optimizer_params['momentum'] = momentum

    # run fit net, every n epochs we run evaluation network to get mAP
    if voc07_metric:
        valid_metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=4)
    else:
        valid_metric = MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=4)

    # random_aspect_exp=2.0, random_aspect_epoch=5, img_stride=32,
    # training with random aspect ratio
    begin_epochs = range(begin_epoch, end_epoch, random_aspect_epoch)
    end_epochs = begin_epochs[1:] + [end_epoch]

    for be, ee in zip(begin_epochs, end_epochs):
        #
        rand_asp = np.sqrt(np.power(random_aspect_exp, np.random.uniform(-1, 1)))
        dh = data_shape[1] / rand_asp
        dh = int(np.round(dh / img_stride)) * img_stride
        dw = data_shape[2] * rand_asp
        dw = int(np.round(dw / img_stride)) * img_stride

        rand_shape = (data_shape[0], dh, dw)
        logger.info('Setting random shape: ({}, {}, {})'.format(dh, dw, data_shape[0]))

        # init iterator
        patch_size = max(data_shape)
        min_gt_scale = min_obj_size / float(patch_size)
        rand_scaler = RandScaler((dw, dh), min_gt_scale=min_gt_scale, force_resize=False)
        train_iter = DetIter(imdb, batch_size, rand_shape[1:], rand_scaler,
                             mean_pixels=mean_pixels, rand_mirror=cfg.train['rand_mirror_prob'] > 0,
                             shuffle=cfg.train['shuffle'], rand_seed=cfg.train['seed'],
                             is_train=True)
        if val_imdb:
            rand_scaler = RandScaler((dw, dh), no_random=True, force_resize=False)
            val_iter = DetIter(val_imdb, batch_size, rand_shape[1:], rand_scaler,
                               mean_pixels=mean_pixels, is_train=True)
        else:
            val_iter = None

        # more informatic parameter setting
        if not mod.binded:
            mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
            mod = set_mod_params(mod, args, auxs, logger)
            mod.init_optimizer(optimizer=optimizer_name, optimizer_params=optimizer_params, force_init=True)
            allow_missing = True

        mod.fit(train_iter,
                eval_data=val_iter,
                eval_metric=MultiBoxMetric(),
                validation_metric=valid_metric,
                batch_end_callback=batch_end_callback,
                epoch_end_callback=epoch_end_callback,
                optimizer=optimizer_name,
                optimizer_params=optimizer_params,
                begin_epoch=be,
                num_epoch=ee,
                initializer=mx.init.Xavier(),
                arg_params=args,
                aux_params=auxs,
                allow_missing=allow_missing,
                monitor=monitor,
                force_rebind=True,
                force_init=True)

        args, auxs = mod.get_params()
        allow_missing = False
