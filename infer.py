# -*- coding: utf-8 -*-
from __future__ import print_function

#import glob
from glob import glob
import os
import argparse
import time

import cv2

import shutil
import numpy as np
from PIL import Image

from detector import Detector
detector = Detector('./ctpn/checkpoints','./ctpn/data/text.yml')
from recognizer import Recognizer
recoer = Recognizer('./crnn/labels/char_std_5990.txt', './crnn/models/weights_densenet.h5')

image_files = glob('./samples/*.*')

def process(img):
    start_time = time.time()
    rois, _, img = detector.detect(img)
    print("CTPN time: %.03fs" % (time.time() - start_time))
    from utilis import sort_box
    rois = sort_box(rois)
    start_time = time.time()
    ocr_result = recoer.recognize(img,rois)
    print("CRNN time: %.03fs" % (time.time() - start_time))
    from idcard import ocr_result2idcard_json
    res = ocr_result2idcard_json(ocr_result)

if __name__ == '__main__':
    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))
        process(image)
