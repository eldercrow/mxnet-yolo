import sys, os
import argparse
import mxnet as mx
import numpy as np

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from tools.prepare_dataset import *


def clsuter_anchor(imdb, box_shapes, n_cluster, data_shape, th_iou=0.5):
    '''
    imdb: imdb that contains labels we will use.
    box_shapes:
        (n, 2) array.
        Each row contains (width, height) of an anchor box.
        n > n_cluster.
    '''
    labels_all = imdb.dump_bb()
    valid_labels = []
    for label in labels_all:
        valid = _get_valid_label(label)
        valid_labels.extend(valid.tolist())
    labels_all = np.array(valid_labels)
    n_label = labels_all.shape[0]

    if box_shapes is None:
        box_shapes = np.random.uniform(0.001, 1, (2000, 2))
    n_box = box_shapes.shape[0]

    wh_label = np.zeros((n_label, 2))
    wh_label[:, 0] = labels_all[:, 3] - labels_all[:, 1]
    wh_label[:, 1] = labels_all[:, 4] - labels_all[:, 2]

    n_cls_samples = [np.sum(labels_all[:, 0] == i) for i in range(imdb.num_classes)]
    cls_weights = [imdb.num_images / float(n) for n in n_cls_samples]
    cls_weights = np.array([n / sum(cls_weights) for n in cls_weights])

    # mimic random augmentation
    # wh_rand = np.zeros((10, n_label, 2))
    # wh_rand[0] = wh_label
    # for i in range(1, 10):
    #     wh_rand[i] = wh_label * np.power(2.0, np.random.uniform(-1.0, 1.0, (n_label, 2)))
    # wh_label = np.reshape(wh_rand, (-1, 2))
    # n_label = wh_label.shape[0]
    # import ipdb
    # ipdb.set_trace()

    # initial grouping
    iou_bb = np.zeros((n_label, n_box))
    for i, bb in enumerate(box_shapes):
        iou_bb[:, i] = _compute_iou(bb, wh_label)
    iou_bb = iou_bb >= th_iou
    n_in = np.sum(iou_bb, axis=0)

    iidx = n_in >= 10
    iou_bb = iou_bb[:, iidx]
    n_box = iou_bb.shape[1]

    n_in = np.sum(np.any(iou_bb, axis=1))
    print 'total number of boxes = {}'.format(n_label)
    print 'number of boxes could be captured before clustering = {}'.format(n_in)

    # J-linkage clustering
    jd_map = np.ones((n_box, n_box))
    for i in range(n_box):
        for j in range(i, n_box):
            jd_map[i, j] = _jaccard_dist(iou_bb[:, i], iou_bb[:, j])
            jd_map[i, i] = 1

    iou_bb0 = iou_bb
    n_curr_cluster = n_box
    is_merged = np.zeros((n_box,), dtype=bool)
    while n_curr_cluster > n_cluster:
        if np.min(jd_map) == 1:
            import ipdb
            ipdb.set_trace()
            break
        midx = np.argmin(jd_map)
        iy, ix = np.unravel_index(midx, dims=jd_map.shape)
        # merge ix to iy
        iou_bb[:, iy] = np.logical_or(iou_bb[:, iy], iou_bb[:, ix])
        iou_bb[:, ix] = 0
        is_merged[ix] = True
        # update jd_map
        for i in range(0, iy):
            jd_map[i, iy] = _jaccard_dist(iou_bb[:, i], iou_bb[:, iy])
        for i in range(iy, n_box):
            jd_map[iy, i] = _jaccard_dist(iou_bb[:, i], iou_bb[:, iy])
        jd_map[iy, iy] = 1
        jd_map[ix, :] = 1
        jd_map[:, ix] = 1
        n_curr_cluster -= 1

    # finished clustering, get initial cluster centers
    cidx = np.where(is_merged == False)[0]
    n_cluster = len(cidx)

    cls_labels = labels_all[:, 0].astype(int)

    cshapes = np.zeros((len(cidx), 2))
    for i, c in enumerate(cidx):
        wh_cluster = wh_label[iou_bb[:, c], :]
        cshapes[i, :] = _update_bb(box_shapes[c, :], wh_cluster, None, None)

    # k-means refinement
    prev_midx = None
    for k in range(500):
        iou_cluster = np.zeros((n_label, n_cluster))
        for i, bb in enumerate(cshapes):
            iou_cluster[:, i] = _compute_iou(bb, wh_label)
        midx = np.argmax(iou_cluster, axis=1)
        if prev_midx is not None and np.all(midx == prev_midx):
            print 'k-means converged in {} iterations.'.format(k)
            break
        prev_midx = midx.copy()

        for i in range(n_cluster):
            wh_cluster = wh_label[midx == i, :]
            cidx = cls_labels[midx == i]
            cw_cluster = cls_weights[cidx]
            cshapes[i, :] = _update_bb(cshapes[i, :], wh_cluster, iou_cluster[midx == i, i], cw_cluster)

    max_iou = np.max(iou_cluster, axis=1)
    mean_iou = np.mean(max_iou)

    per_cls_cshapes = _per_class_refinement(cshapes, labels_all[:, 0], wh_label, imdb.num_classes)

    for i, bb in enumerate(cshapes):
        iou_cluster[:, i] = _compute_iou(bb, wh_label)
    iou_cluster = iou_cluster >= 0.5
    n_in = np.sum(np.any(iou_cluster, axis=1))
    print 'number of boxes could be captured by clusters = {}'.format(n_in)

    csr = np.zeros_like(cshapes)
    csr[:, 0] = np.sqrt(cshapes[:, 0] * cshapes[:, 1])
    csr[:, 1] = cshapes[:, 0] / cshapes[:, 1]
    sidx = np.argsort(csr[:, 0])
    csr = csr[sidx, :]
    cshapes = cshapes[sidx, :] * data_shape

    np.set_printoptions(suppress=True)

    print 'cluster centers after k-means.'
    print np.round(cshapes, 3)

    cratio = np.round(cshapes / 16.0, 3)
    print 'divided by 16, for anchor layer parameter.'
    for c in cratio:
        print '\t{}, {},'.format(c[0], c[1])
    # print 'squared cluster size and ratio.'
    # print np.round(csr, 3)
    print 'mean IOU = {}.'.format(mean_iou)


def _update_bb(bb, wh_cluster, iou_cluster, cw_cluster):
    #
    return np.exp(np.mean(np.log(wh_cluster), axis=0))
    # if iou_cluster is None:
    #     return np.mean(wh_cluster, axis=0) #
    # ix = np.exp(-np.abs(np.log(bb[0] / wh_cluster[:, 0])))
    # iy = np.exp(-np.abs(np.log(bb[1] / wh_cluster[:, 1])))
    # diff_x = wh_cluster[:, 0] * (1 - ix) * cw_cluster
    # diff_y = wh_cluster[:, 1] * (1 - iy) * cw_cluster
    # bb[0] = np.mean(diff_x) / np.mean((1 - ix) * cw_cluster)
    # bb[1] = np.mean(diff_y) / np.mean((1 - iy) * cw_cluster)
    # return bb


def _get_valid_label(labels):
    #
    vidx = np.any(labels != -1, axis=1)
    return labels[vidx, :]


def _compute_iou(bb, wh_label):
    ix = np.exp(-np.abs(np.log(wh_label[:, 0] / bb[0])))
    iy = np.exp(-np.abs(np.log(wh_label[:, 1] / bb[1])))
    return ix * iy


def _jaccard_dist(lhs, rhs):
    #
    uu = np.sum(np.logical_or(lhs, rhs))
    if uu == 0:
        return 1
    ii = np.sum(np.logical_and(lhs, rhs))
    return (uu - ii) / float(uu)


def _per_class_refinement(cshapes, cls_label, wh_label, n_class):
    #
    n_label = wh_label.shape[0]
    n_cluster = cshapes.shape[0]

    iou_cluster = np.zeros((n_label, n_cluster))
    for i, bb in enumerate(cshapes):
        iou_cluster[:, i] = _compute_iou(bb, wh_label)

    # label to cluster shape mapping
    midx = np.argmax(iou_cluster, axis=1)

    cluster_ids = [np.where(midx == i)[0] for i in range(n_cluster)]
    class_ids = [np.where(cls_label == i)[0] for i in range(n_class)]

    #
    # per_cls_ratio = np.zeros((n_class, 2))
    # for i in range(n_class):
    #     cls_cshapes = cshapes[midx[class_ids[i]], :]
    #     cls_wh = wh_label[class_ids[i], :]
    #
    #     per_cls_ratio[i] = np.exp(np.mean(np.log(cls_wh / cls_cshapes), axis=0))
    # return per_cls_ratio
    #
    per_cls_shapes = np.zeros((n_cluster, n_class, 2))
    for i in range(n_cluster):
        # for each cluster
        cshape = cshapes[i]

        for j in range(n_class):
            idx = np.intersect1d(cluster_ids[i], class_ids[j])
            if idx.size == 0:
                per_cls_shapes[i, j, :] = cshape
            else:
                cshape = np.reshape(cshape, (1, -1))
                per_cls_shapes[i, j, :] = np.exp(np.mean(np.log(wh_label[idx, :] / cshape), axis=0))
                # per_cls_shapes[i, j, :] = _update_bb(cshape, wh_label[idx, :])

    return per_cls_shapes
    #


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--dataset', dest='dataset', help='dataset to use',
                        default='pascal', type=str)
    parser.add_argument('--year', dest='year', help='which year to use',
                        default='2007,2012', type=str)
    parser.add_argument('--set', dest='set', help='train, val, trainval, test',
                        default='trainval', type=str)
    parser.add_argument('--root', dest='root_path', help='dataset root path',
                        default=os.path.join(curr_path, '..', 'data', 'VOCdevkit'),
                        type=str)
    parser.add_argument('--num-cluster', dest='num_cluster', help='number of cluster',
                        default=5, type=int)
    parser.add_argument('--data-shape', dest='data_shape', help='data (image) shape',
                        default=416, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    sizes = np.arange(1, 21) / 20.0
    sizes = np.power(2.0, sizes) - 1.0
    ratios = [1.0, 0.5, 0.3333, 0.25, 2.0, 3.0, 4.0]
    ratios = np.array(ratios)

    shapes = np.zeros((len(sizes)*len(ratios), 2))
    k = 0
    for s in sizes:
        for r in ratios:
            r2 = np.sqrt(r)
            shapes[k, 0] = s * r2
            shapes[k, 1] = s / r2
            k += 1

    if args.dataset == 'pascal':
        imdb = load_pascal(args.set, args.year, args.root_path, False)
        clsuter_anchor(imdb, shapes, args.num_cluster, args.data_shape)
    else:
        raise NotImplementedError("No implementation for dataset: " + args.dataset)
