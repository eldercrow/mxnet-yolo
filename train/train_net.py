import tools.find_mxnet
import mxnet as mx
import logging
import sys
import os
import importlib
import re
import random
from dataset.iterator import DetRecordIter
from train.metric import MultiBoxMetric
from evaluate.eval_metric import MApMetric, VOC07MApMetric
from config.config import cfg
from train.lr_scheduler import BurnInMultiFactorScheduler
from symbol.symbol_builder import get_symbol_train

def convert_pretrained(name, args):
    """
    Special operations need to be made due to name inconsistance, etc

    Parameters:
    ---------
    name : str
        pretrained model name
    args : dict
        loaded arguments

    Returns:
    ---------
    processed arguments as dict
    """
    if 'vgg16_reduced' in name:
        args['conv6_bias'] = args.pop('fc6_bias')
        args['conv6_weight'] = args.pop('fc6_weight')
        args['conv7_bias'] = args.pop('fc7_bias')
        args['conv7_weight'] = args.pop('fc7_weight')
        del args['fc8_weight']
        del args['fc8_bias']
    return args

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
        lr_scheduler = BurnInMultiFactorScheduler(burn_in=1000, step=steps, factor=lr_refactor_ratio)
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

def train_net(net, train_path, num_classes, batch_size,
              data_shape, mean_pixels, resume, finetune, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate,
              momentum, weight_decay, lr_refactor_step, lr_refactor_ratio,
              freeze_layer_pattern='',
              shape_range=(320, 512), random_shape_step=0, random_shape_epoch=10,
              num_example=10000, label_pad_width=350,
              nms_thresh=0.45, force_nms=False, ovp_thresh=0.5,
              use_difficult=False, class_names=None,
              voc07_metric=False, nms_topk=400, force_suppress=False,
              train_list="", val_path="", val_list="", iter_monitor=0,
              monitor_pattern=".*", log_file=None):
    """
    Wrapper for training phase.

    Parameters:
    ----------
    net : str
        symbol name for the network structure
    train_path : str
        record file path for training
    num_classes : int
        number of object classes, not including background
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
    shape_range : tuple of (min, max)
        random data shape range
    random_shape_step : int
        step size for random data shape, defined by network, 0 to disable
    random_step_epoch : int
        number of epoch before next random shape
    num_example : int
        number of training images
    label_pad_width : int
        force padding training and validation labels to sync their label widths
    nms_thresh : float
        non-maximum suppression threshold for validation
    force_nms : boolean
        suppress overlaped objects from different classes
    train_list : str
        list file path for training, this will replace the embeded labels in record
    val_path : str
        record file path for validation
    val_list : str
        list file path for validation, this will replace the embeded labels in record
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

    # load symbol
    sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    # symbol_module = importlib.import_module("symbol_" + net)
    # net = symbol_module.get_symbol(num_classes, nms_thresh=nms_thresh,
    #     force_suppress=force_suppress, nms_topk=nms_topk)
    net = get_symbol_train(net, num_classes, nms_thresh, force_suppress, nms_topk)

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
    elif finetune > 0:
        logger.info("Start finetuning with {} from epoch {}"
            .format(ctx_str, finetune))
        _, args, auxs = mx.model.load_checkpoint(prefix, finetune)
        begin_epoch = finetune
        # the prediction convolution layers name starts with relu, so it's fine
        fixed_param_names = [name for name in net.list_arguments() \
            if name.startswith('conv')]
    elif pretrained:
        logger.info("Start training with {} from pretrained model {}"
            .format(ctx_str, pretrained))
        _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
        args = convert_pretrained(pretrained, args)
    else:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None

    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # fit parameters
    batch_end_callback = mx.callback.Speedometer(batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None

    # run fit net, every n epochs we run evaluation network to get mAP
    if voc07_metric:
        valid_metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=4)
    else:
        valid_metric = MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=4)

    # init training module
    mod = mx.mod.Module(net, label_names=('yolo_output_label',), logger=logger, context=ctx,
                        fixed_param_names=fixed_param_names)

    random_shape_step = int(random_shape_step)
    if random_shape_step > 0:
        fit_begins = range(begin_epoch, end_epoch, random_shape_epoch)
        fit_ends = fit_begins[1:] + [end_epoch]
        assert (len(shape_range) == 2)
        data_shapes = [(3, x * random_shape_step, x * random_shape_step) \
            for x in range(shape_range[0] // random_shape_step,
            shape_range[1] // random_shape_step + 1)]
        logger.info("Candidate random shapes:" + str(data_shapes))
    else:
        fit_begins = [begin_epoch]
        fit_ends = [end_epoch]
        data_shapes = [data_shape]

    for begin, end in zip(fit_begins, fit_ends):
        if len(data_shapes) == 1:
            data_shape = data_shapes[0]
        else:
            data_shape = data_shapes[random.randint(0, len(data_shapes)-1)]
            logger.info("Setting random data shape: " + str(data_shape))

        train_iter = DetRecordIter(train_path, batch_size, data_shape, mean_pixels=mean_pixels,
            label_pad_width=label_pad_width, path_imglist=train_list, **cfg.train)

        if val_path:
            val_iter = DetRecordIter(val_path, batch_size, data_shape, mean_pixels=mean_pixels,
                label_pad_width=label_pad_width, path_imglist=val_list, **cfg.valid)
        else:
            val_iter = None

        learning_rate, lr_scheduler = get_lr_scheduler(learning_rate, lr_refactor_step,
            lr_refactor_ratio, num_example, batch_size, begin_epoch)
        # optimizer_params={'learning_rate':learning_rate,
        #                   'wd':weight_decay,
        #                   'lr_scheduler':lr_scheduler,
        #                   'clip_gradient':10,
        #                   'rescale_grad': 1.0 }
        optimizer_params={'learning_rate':learning_rate,
                          'momentum':momentum,
                          'wd':weight_decay,
                          'lr_scheduler':lr_scheduler,
                          'clip_gradient':10,
                          'rescale_grad': 1.0 }

        # more informatic parameter setting
        if not mod.binded:
            mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
            mod = set_mod_params(mod, args, auxs, logger)

        mod.fit(train_iter,
                val_iter,
                eval_metric=MultiBoxMetric(),
                validation_metric=valid_metric,
                batch_end_callback=batch_end_callback,
                epoch_end_callback=epoch_end_callback,
                optimizer='sgd',
                optimizer_params=optimizer_params,
                begin_epoch=begin,
                num_epoch=end,
                initializer=mx.init.Xavier(),
                arg_params=args,
                aux_params=auxs,
                allow_missing=allow_missing,
                monitor=monitor,
                force_rebind=True,
                force_init=True)

        args, auxs = mod.get_params()
        allow_missing = False
