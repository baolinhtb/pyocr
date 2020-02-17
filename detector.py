#!/usr/env/bin python3

import os
import sys
import math
import numpy as np
import tensorflow as tf

from helper import utils

from utils import resize_im, draw_boxes

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'ctpn', 'lib')
sys.path.insert(0, lib_path)
from ctpn.lib.utils.timer import Timer
from ctpn.lib.fast_rcnn.config import cfg
from ctpn.lib.fast_rcnn.test import  test_ctpn
from ctpn.lib.networks.factory import get_network
from ctpn.lib.text_connector.detectors import TextDetector
from ctpn.lib.text_connector.text_connect_cfg import Config as TextLineCfg
from ctpn.lib.fast_rcnn.config import cfg_from_file

class Detector:

    def __init__(self, ckpt,config='./ctpn/data/text.yml'):
        """
        :param ckpt: ckpt 目录或者 pb 文件
        """
        cfg_from_file(config)
        self.textdetector = TextDetector()
        self.sess, self.graph = self.load_ckpt(ckpt)
        

    def load_ckpt(self,model_path='./ctpn/checkpoints'):
        # load config file
        cfg.TEST.checkpoints_path = model_path

        # init session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        sess = tf.Session(config=config)

        # load network
        net = get_network("VGGnet_test")

        # load model
        print('Loading network {:s}... '.format("VGGnet_test"))
        saver = tf.train.Saver()
        try:
            ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('done')
        except:
            raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        return sess, net

    def detect(self, img):
        """
        :param img: RGB
        :return:  text_lines point order: left-top, right-top, left-bottom, right-bottom
        """
        timer = Timer()
        timer.tic()

        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(self.sess, self.graph, img)

        boxes = self.textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        timer.toc()
        print("\n----------------------------------------------")
        print(('Detection took {:.3f}s for '
            '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
        text_recs, img_drawed = draw_boxes(img, boxes, scale)
        return text_recs, img_drawed, img

    def recover_scale(self, boxes, scale):
        """
        :param boxes: [(x1, y1, x2, y2)]
        :param scale: image scale
        :return:
        """
        tmp_boxes = []
        for b in boxes:
            tmp_boxes.append([int(x / scale) for x in b])
        return np.asarray(tmp_boxes)

    def get_line_boxes(self, boxes, scale=1):
        """
        Get bounding boxes from four point
        :param boxes: (x1, y1, x2, y2, x3, y3, x4, y4)
        :param scale: scale returned by resize_im
        :return
            [(min_x, min_y, max_x, max_y), ...]
        """
        ret = []
        for box in boxes:
            min_x = min(int(box[0] / scale), int(box[2] / scale),
                        int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale),
                        int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale),
                        int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale),
                        int(box[5] / scale), int(box[7] / scale))

            ret.append([min_x, min_y, max_x, max_y])

        return ret


if __name__ == "__main__":
    detector = Detector('./ctpn/checkpoints')
