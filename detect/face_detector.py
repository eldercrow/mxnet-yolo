from __future__ import print_function
import os
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.face_test_iter import FaceTestIter
from tools.load_checkpoint import load_checkpoint

class FaceDetector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """

    def __init__(self, symbol, model_prefix, epoch, data_hw, mean_pixels,
                 img_stride=32, th_nms=0.45, ctx=None):
        '''
        '''
        self.ctx = mx.cpu() if not ctx else ctx

        if isinstance(data_hw, int):
            data_hw = (data_hw, data_hw)
        assert data_hw[0] % img_stride == 0 and data_hw[1] % img_stride == 0
        self.data_hw = data_hw

        arg_params, aux_params = load_checkpoint(model_prefix, epoch)

        self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
        self.mod.bind(data_shapes=[('data', (1, 3, data_hw[0], data_hw[1]))])
        self.mod.set_params(arg_params, aux_params)

        self.mean_pixels = mean_pixels
        self.img_stride = img_stride
        self.th_nms = th_nms

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size

        result = []
        im_paths = []
        detections = []
        time_elapsed = 0
        for i, (datum, im_info) in enumerate(det_iter):
            im_paths.append(im_info['im_path'])
            self.mod.reshape(data_shapes=datum.provide_data)

            start = timer()
            self.mod.forward(datum)
            out = self.mod.get_outputs()
            det = out[0][0].asnumpy()
            det = self._do_nms(det, self.th_nms)
            pidx = np.where(det[:, 0] >= 0)[0]
            det = det[pidx, :]
            # sidx = np.argsort(det[:, 1])[::-1]
            # det = det[sidx, :]
            # vidx = self._do_nms(det)
            # det = det[vidx, :]
            time_elapsed += timer() - start

            result.append(det)

            if i % 10 == 0:
                n_dets = det.shape[0]
                print('Processing image {}/{}, {} objects detected.'.format(i+1, num_images, n_dets))
        # time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(num_images, time_elapsed))
        return result, im_paths

    def im_detect(self,
                  im_list,
                  root_dir=None,
                  extension=None,
                  show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = FaceTestIter(test_db,
                min_hw=self.data_hw, mean_pixels=self.mean_pixels, img_stride=self.img_stride)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        colors = dict()
        hh, ww, _ = img.shape
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * ww)
                    ymin = int(dets[i, 3] * hh)
                    xmax = int(dets[i, 4] * ww)
                    ymax = int(dets[i, 5] * hh)
                    rect = plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        edgecolor=colors[cls_id],
                        linewidth=1.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    # plt.gca().text(
                    #     xmin,
                    #     ymin - 2,
                    #     '{:.3f}'.format(score),
                    #     bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    #     fontsize=7,
                    #     color='white')
                    plt.gca().text(
                        xmin,
                        ymin - 2,
                        '{:s} {:.3f}'.format(class_name, score),
                        bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                        fontsize=7,
                        color='white')
        # plt.gcf().savefig('res.png')
        plt.show()

    def detect_and_visualize(self,
                             im_list,
                             root_dir=None,
                             extension=None,
                             classes=[],
                             thresh=0.6,
                             show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        import cv2
        dets, _ = self.im_detect(
            im_list, root_dir, extension, show_timer=show_timer)
        root_dir = '' if not root_dir else root_dir
        extension = '' if not extension else extension
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            fn_img = os.path.join(root_dir, im_list[k] + extension)
            img = cv2.imread(fn_img)
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection(img, det, classes, thresh)


    def _do_nms(self, dets, th_nms=0.7):
        #
        areas = (dets[:, 4] - dets[:, 2]) * (dets[:, 5] - dets[:, 3])
        vmask = np.ones((dets.shape[0],), dtype=int)
        vidx = []
        for i, d in enumerate(dets):
            if vmask[i] == 0:
                continue
            iw = np.minimum(d[4], dets[i:, 4]) - np.maximum(d[2], dets[i:, 2])
            ih = np.minimum(d[5], dets[i:, 5]) - np.maximum(d[3], dets[i:, 3])
            I = np.maximum(iw, 0) * np.maximum(ih, 0)
            iou = I / np.maximum(areas[i:] + areas[i] - I, 1e-08)
            nidx = np.where(iou > th_nms)[0] + i
            vmask[nidx] = 0
            vidx.append(i)
        return dets[vidx, :]
    #
    #
    # def _comp_overlap(self, dets, im_shape):
    #     #
    #     area_dets = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    #     # if np.min(area_dets) < 16 or np.max(area_dets) > im_shape[0]*im_shape[1]*2.0:
    #     #     import ipdb
    #     #     ipdb.set_trace()
    #     iw = np.minimum(dets[:, 2], im_shape[1]) - np.maximum(dets[:, 0], 0)
    #     ih = np.minimum(dets[:, 3], im_shape[0]) - np.maximum(dets[:, 1], 0)
    #
    #     overlap = (iw * ih) / (area_dets + 1e-04)
    #     return overlap