# Source Generated with Decompyle++
# File: rand_sampler.pyc (Python 2.7)

import numpy as np
import mxnet as mx
import math
import cv2

class RandSampler(object):
    '''
    Random sampler base class, used for data augmentation

    Parameters:
    ----------
    max_trials : int
        maximum trials, if exceed this number, give up anyway
    max_sample : int
        maximum random crop samples to be generated
    '''
    def __init__(self, max_trials, max_sample):
        assert max_trials > 0
        self.max_trials = int(max_trials)
        assert max_sample >= 0
        self.max_sample = int(max_sample)

    def sample(self, label):
        '''
        Interface for calling sampling function

        Parameters:
        ----------
        label : numpy.array (n x 5 matrix)
            ground-truths

        Returns:
        ----------
        list of (crop_box, label) tuples, if failed, return empty list []
        '''
        return NotImplementedError


class RandScaler(RandSampler):
    '''
    Random scaling and cropping original images with various settings
    '''
    def __init__(self, patch_wh, no_random=False,
            min_gt_scale=0.01, max_trials=50, force_resize=False, clip_bb=True):
        '''
        Experimental. Some parameters are hardcoded.
        '''
        super(RandScaler, self).__init__(max_trials, max_sample=1)

        if (min_gt_scale < 0) or (min_gt_scale > 1):
            raise AssertionError('min_gt_scale must in [0, 1]')
        self.min_gt_scale = min_gt_scale

        self.patch_wh = patch_wh
        self.force_resize = force_resize
        self.clip_bb = clip_bb

        self.no_random = no_random
        self.scale_exp = 3
        self.aspect_exp = 2.0 if self.force_resize else 1.25

        self.min_gt_overlap = 0.5
        self.min_gt_ignore = 0.25

    def sample(self, label, img_wh):
        '''
        generate random padding boxes according to parameters
        if satifactory padding generated, apply to ground-truth as well

        Parameters:
        ----------
        label : numpy.array (n x 5 matrix)
            ground-truths

        Returns:
        ----------
        list of (crop_box, label) tuples, if failed, return empty list []
        '''
        valid_mask = np.where(np.all(label == -1, axis = 1) == False)[0]
        gt = label[valid_mask, :5]

        # label in pixel coordinate
        gt[:, 1::2] *= img_wh[0]
        gt[:, 2::2] *= img_wh[1]

        samples = []
        for trial in range(self.max_trials):
            if (not self.no_random) and (trial < (self.max_trials - 1)):
                sf = float(self.patch_wh[0]*self.patch_wh[1]) / float(img_wh[0]*img_wh[1])
                sf = np.sqrt(sf)
                sf *= np.power(self.scale_exp, np.random.uniform(-1, 1))
                asp = np.power(self.aspect_exp, np.random.uniform(-1, 1))
                asp = np.sqrt(asp)

                scale_x = sf * asp
                scale_y = sf / asp

                patch_sz_x = np.round(self.patch_wh[0] / scale_x)
                patch_sz_y = np.round(self.patch_wh[1] / scale_y)

                dx = img_wh[0] - patch_sz_x
                dy = img_wh[1] - patch_sz_y
                if dx != 0:
                    dx = np.random.randint(low=np.minimum(dx, 0), high=np.maximum(dx, 0))
                if dy != 0:
                    dy = np.random.randint(low=np.minimum(dy, 0), high=np.maximum(dy, 0))
            else:
                if self.force_resize:
                    dx, dy, patch_sz_x, patch_sz_y = 0, 0, self.patch_wh[0], self.patch_wh[1]
                else:
                    sf = np.sqrt(float(self.patch_wh[0] * self.patch_wh[1]) / float(img_wh[0] * img_wh[1]))

                    patch_sz_x = np.round(self.patch_wh[0] / sf)
                    patch_sz_y = np.round(self.patch_wh[1] / sf)

                    dx = (img_wh[0] - patch_sz_x) // 2
                    dy = (img_wh[1] - patch_sz_y) // 2

            bbox = [dx, dy, dx + patch_sz_x, dy + patch_sz_y]

            new_gt_boxes = []
            for (i, bb) in enumerate(gt):
                # discard oob gt
                overlap = _compute_overlap(bb[1:], bbox)
                if overlap < self.min_gt_ignore:
                    continue
                l = bb[0]
                if not self.no_random:
                    # discard too small gt
                    new_size = max((bb[4] - bb[2]) / patch_sz_y, (bb[3] - bb[1]) / patch_sz_x)
                    if new_size < self.min_gt_scale:
                        continue
                    # set ignore label to dubious sample
                    if overlap < self.min_gt_overlap:
                        l = -1
                new_gt_boxes.append([l, bb[1]-dx, bb[2]-dy, bb[3]-dx, bb[4]-dy])

            if not (self.no_random) and len(new_gt_boxes) == 0 and trial < self.max_trials - 1:
                continue

            if len(new_gt_boxes) == 0:
                new_gt_boxes.append([-1, 0, 0, patch_sz_x, patch_sz_y])

            new_gt_boxes = np.reshape(np.array(new_gt_boxes), (-1, 5))
            new_gt_boxes[:, 1::2] /= float(patch_sz_x)
            new_gt_boxes[:, 2::2] /= float(patch_sz_y)

            if self.clip_bb:
                new_gt_boxes[:, 1:3] = np.maximum(0, new_gt_boxes[:, 1:3])
                new_gt_boxes[:, 3:5] = np.minimum(1, new_gt_boxes[:, 3:5])

            # if not self.force_resize:
            #     new_gt_boxes[:, 1::2] *= self.patch_wh[0]
            #     new_gt_boxes[:, 2::2] *= self.patch_wh[1]

            label = np.lib.pad(new_gt_boxes, \
                    ((0, label.shape[0]-new_gt_boxes.shape[0]), (0, 0)), \
                    'constant', constant_values = (-1, -1))
            rr = 1.0 #if self.force_resize else self.patch_wh[0]
            samples = (bbox, label, rr)
            break

        return samples


def _compute_overlap(roi, img_roi):
    ox = _compute_overlap_1d(roi[0], roi[2], img_roi[0], img_roi[2])
    oy = _compute_overlap_1d(roi[1], roi[3], img_roi[1], img_roi[3])
    return ox * oy


def _compute_overlap_1d(p0, p1, q0, q1):
    ''' p0, p1, q0, q1: size of (n_rows, ) '''
    I = np.maximum(0, np.minimum(p1, q1) - np.maximum(p0, q0))
    U = np.maximum(1e-08, p1 - p0)
    return np.minimum(1, I / U)


class ColorJitter(object):
    '''
    For random color augmentation.
    '''
    def __init__(self, mean_rgb=(0.5, 0.5, 0.5), \
            prob_h=0.5, prob_l=0.5, prob_s=0.5, prob_c=0.5, \
            max_rand_h = 0.1, max_rand_l = 0.125, max_rand_s = 0.125, max_rand_c = 0.5):
        #
        self.mean_bgr = np.reshape(np.array(mean_rgb)[::-1], (1, 1, 3)) / 255.0
        self.prob_h = prob_h
        self.prob_l = prob_l
        self.prob_s = prob_s
        self.prob_c = prob_c
        self.rand_h = max_rand_h
        self.rand_l = max_rand_l
        self.rand_s = max_rand_s
        self.rand_c = max_rand_c

    def sample(self, data):
        #
        img = data.asnumpy() / 255.0
        img = img[:, :, ::-1].astype(np.float32)

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        (hh, ww, _) = hls.shape

        rands = np.random.uniform(0, 1, 8)
        rv = rands[4:] * 2 - 1

        if rands[0] < self.prob_h:
            h = hls[:, :, 0] + rv[0] * self.rand_h * 360
            h = np.mod(h, 360)

        if rands[1] < self.prob_l:
            hls[:, :, 1] += rv[1] * self.rand_l
            hls[:, :, 1] = np.maximum(np.minimum(hls[:, :, 1], 1), 0)

        if rands[2] < self.prob_s:
            hls[:, :, 2] += rv[2] * self.rand_s
            hls[:, :, 2] = np.maximum(np.minimum(hls[:, :, 2], 1), 0)

        img = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

        if rands[3] < self.prob_c:
            rc = np.power(1 + self.rand_c, rv[3])
            img = (img - self.mean_bgr) * rc + self.mean_bgr
            img = np.minimum(np.maximum(img, 0), 1)

        img = img[:, :, ::-1]
        return mx.nd.array(img * 255, dtype=data.dtype)
