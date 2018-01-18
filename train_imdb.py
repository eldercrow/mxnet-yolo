import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
from train.train_net import train_net
from dataset.dataset_loader import load_pascal

def parse_args():
    #
    parser = argparse.ArgumentParser(description='Train a YOLO detection network')
    #
    parser.add_argument('--dataset', dest='dataset', help='which dataset to use, check .dataset directory',
                        default='pascal_voc', type=str)
    parser.add_argument('--image-set', dest='image_set', help='train set, can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2012',
                        default='2007,2012', type=str)
    parser.add_argument('--val-image-set', dest='val_image_set', help='validation set, can be val or test',
                        default='', type=str)
    parser.add_argument('--val-year', dest='val_year', help='can be 2007, 2012',
                        default='', type=str)
    parser.add_argument('--devkit-path', dest='devkit_path', help='VOCdevkit or Wider path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--min-obj-size', dest='min_obj_size', type=float, default=8.0,
                        help='minimum object size to be used for training.')
    parser.add_argument('--network', dest='network', type=str, default='mobilenet_yolo',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1,
                        help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'darknet19'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'yolo2'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=150, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=576,
                        help='set image shape')
    parser.add_argument('--random-aspect-exp', dest='random_aspect_exp', type=float,
                        default=2.0, help='random data shape step')
    parser.add_argument('--random-aspect-epoch', dest='random_aspect_epoch', type=int,
                        default=5, help='random shape epoch')
    parser.add_argument('--img-stride', dest='img_stride', type=int,
                        default=32, help='image stride')
    parser.add_argument('--optimizer-name', dest='optimizer_name', type=str, default='nadam',
                        help='optimizer name')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123.68,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=116.779,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=103.939,
                        help='blue mean value')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='80,120',
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=float, default=0.1,
                        help='ratio to refactor learning rate')
    parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="^(conv1_|conv2_).*",
                        help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='save training log to file')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    parser.add_argument('--pattern', dest='monitor_pattern', type=str, default=".*",
                        help='monitor parameter pattern, as regex')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.35,
                        help='non-maximum suppression threshold')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=True,
                        help='use PASCAL VOC 07 11-point metric')
    args = parser.parse_args()
    return args


def load_imdb(dataset, image_set, devkit_path='', year=2007, shuffle=False):
    #
    # for now pascal voc only
    if dataset == 'pascal_voc':
        imdb = load_pascal(image_set, year, devkit_path, shuffle)
    else:
        raise NotImplementedError("Dataset " + dataset + " not supported")
    return imdb

if __name__ == '__main__':
    #
    args = parse_args()
    #
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    imdb = load_imdb(args.dataset, args.image_set, args.devkit_path, args.year, cfg.train['shuffle'])
    val_imdb = None
    if args.val_image_set:
        val_imdb = load_imdb(args.dataset, args.val_image_set, args.devkit_path, args.val_year, False)

    # start training
    train_net(args.network, imdb,
              args.batch_size, args.data_shape, [args.mean_r, args.mean_g, args.mean_b],
              args.resume, args.finetune, args.pretrained,
              args.epoch, args.prefix, ctx, args.begin_epoch, args.end_epoch,
              args.frequent,
              args.optimizer_name, args.learning_rate, args.momentum, args.weight_decay,
              args.lr_refactor_step, args.lr_refactor_ratio,
              val_imdb=val_imdb,
              freeze_layer_pattern=args.freeze_pattern,
              iter_monitor=args.monitor,
              monitor_pattern=args.monitor_pattern,
              log_file=args.log_file,
              min_obj_size=args.min_obj_size,
              use_difficult=args.use_difficult,
              random_aspect_exp=args.random_aspect_exp,
              random_aspect_epoch=args.random_shape_epoch,
              img_stride=args.img_stride,
              nms_thresh=args.nms_thresh,
              ovp_thresh=args.overlap_thresh,
              force_suppress=args.force_nms,
              voc07_metric=args.use_voc07_metric)
              # args.num_class, args.batch_size,
              # args.data_shape, [args.mean_r, args.mean_g, args.mean_b],
              # args.resume, args.finetune, args.pretrained,
              # args.epoch, args.prefix, ctx, args.begin_epoch, args.end_epoch,
              # args.frequent, args.learning_rate, args.momentum, args.weight_decay,
              # args.lr_refactor_step, args.lr_refactor_ratio,
              # val_path=args.val_path,
              # num_example=args.num_example,
              # class_names=class_names,
              # label_pad_width=args.label_width,
              # optimizer_name=args.optimizer_name,
              # freeze_layer_pattern=args.freeze_pattern,
              # iter_monitor=args.monitor,
              # monitor_pattern=args.monitor_pattern,
              # log_file=args.log_file,
              # nms_thresh=args.nms_thresh,
              # force_nms=args.force_nms,
              # ovp_thresh=args.overlap_thresh,
              # use_difficult=args.use_difficult,
              # voc07_metric=args.use_voc07_metric,
              # random_shape_step=args.random_shape_step,
              # shape_range=(args.min_random_shape, args.max_random_shape),
              # random_shape_epoch=args.random_shape_epoch)
