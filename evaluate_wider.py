import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
from evaluate.evaluate_net_wider import evaluate_net
from dataset.dataset_loader import load_pascal

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a network')
    parser.add_argument('--dataset', dest='dataset', help='which dataset to use, check .dataset directory',
                        default='pascal_voc', type=str)
    parser.add_argument('--image-set', dest='image_set', help='train set, can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2012',
                        default='2007,2012', type=str)
    parser.add_argument('--devkit-path', dest='devkit_path', help='VOCdevkit or Wider path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='evaluation batch size')
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='load model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd_'), type=str)
    parser.add_argument('--gpus', dest='gpu_id', help='GPU devices to evaluate with',
                        default='0', type=str)
    parser.add_argument('--cpu', dest='cpu', help='use cpu to evaluate, this can be slow',
                        action='store_true')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=576,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=True,
                        help='use PASCAL VOC 07 metric')
    parser.add_argument('--deploy', dest='deploy_net', help='Load network from model',
                        action='store_true', default=False)
    args = parser.parse_args()
    return args


def load_imdb(dataset, image_set, devkit_path='', year=2007, shuffle=False):
    #
    if dataset == 'pascal_voc':
        imdb = load_pascal(image_set, year, devkit_path, shuffle)
    # elif dataset == 'wider':
    #     imdb = load_wider(image_set, devkit_path, shuffle)
    # elif dataset == 'mscoco':
    #     imdb = load_mscoco(image_set, devkit_path, shuffle)
    else:
        raise NotImplementedError("Dataset " + dataset + " not supported")

    return imdb


if __name__ == '__main__':
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    args = parse_args()
    # choose ctx
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpu_id.split(',')]

    imdb = load_imdb(args.dataset, args.image_set, args.devkit_path, args.year, False)

    network = None if args.deploy_net else args.network
    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network
    else:
        prefix = args.prefix

    evaluate_net(network, imdb,
                 (args.mean_r, args.mean_g, args.mean_b), args.data_shape,
                 prefix, args.epoch, ctx, batch_size=args.batch_size,
                 nms_thresh=args.nms_thresh, force_nms=args.force_nms, ovp_thresh=args.overlap_thresh,
                 use_difficult=args.use_difficult,
                 voc07_metric=args.use_voc07_metric)
