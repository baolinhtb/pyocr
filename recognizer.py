#!/usr/env/bin python3

import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps
# from helper import utils
from math import degrees,atan2

from utils import sort_box,dumpRotateImage
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'crnn')
sys.path.insert(0, lib_path)
from keras.layers import Input
from keras.models import Model
from crnn import keys
from crnn import densenet
from crnn.label_converter import LabelConverter
from imp import reload
reload(densenet)

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

class Recognizer:
    #
    def __init__(self, chars_file='./crnn/labels/char_std_5990.txt', \
                       model_path='./crnn/models/weights_densenet.h5'):
        """
        :param ckpt: ckpt 目录或者 pb 文件
        """
        self.converter = LabelConverter(chars_file)
        input = Input(shape=(32, None, 1), name='the_input')
        y_pred= densenet.dense_cnn(input, nclass)
        self.basemodel = Model(inputs=input, outputs=y_pred)

        # model_path = os.path.join(os.getcwd(), model_path)
        if os.path.exists(model_path):
            self.basemodel.load_weights(model_path)

    def decode(self, predicts):
        decoded_predicts = self.converter.decode_list(predicts, invalid_index=-1)
        ocr_results = [''.join(x) for x in decoded_predicts]
        return ocr_results

    def predict(self,img):
        width, height = img.size[0], img.size[1]
        scale = height * 1.0 / 32
        width = int(width / scale)
        
        img = img.resize([width, 32], Image.ANTIALIAS)
    
        '''
        img_array = np.array(img.convert('1'))
        boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
        if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
            img = ImageOps.invert(img)
        '''

        img = np.array(img).astype(np.float32) / 255.0 - 0.5
        
        X = img.reshape([1, 32, width, 1])
        
        y_pred = self.basemodel.predict(X)
        y_pred = y_pred[:, :, :]

        # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
        # out = u''.join([characters[x] for x in out[0]])
        out = decode(y_pred)

        return out



    def recognize(self, img, text_recs, adjust=False):
        """
        加载OCR模型，进行字符识别
        """
        results = {}
        xDim, yDim = img.shape[1], img.shape[0]
            
        for index, rec in enumerate(text_recs):
            xlength = int((rec[6] - rec[0]) * 0.1)
            ylength = int((rec[7] - rec[1]) * 0.2)
            if adjust:
                pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
                pt4 = (rec[4], rec[5])
            else:
                pt1 = (max(1, rec[0]), max(1, rec[1]))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
                pt4 = (rec[4], rec[5])
                
            degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

            partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)

            if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
                continue

            image = Image.fromarray(partImg).convert('L')
            text = self.predict(image)
            
            if len(text) > 0:
                results[index] = [rec]
                results[index].append(text)  # 识别文字
        
        return results

    # def remove_padding(self, ocr_results):
    #     # roi 图片 padding 以后，识别结果中末尾会有多余的字符，目前多余的字符都是 `;` ，这里临时性地将它移除
    #     out = []

    #     for result in ocr_results:
    #         r = result.rstrip(';；')
    #         out.append(r)
    #     return out

    # def get_batch_imgs(self, img_rois):
    #     max_width = max(img_rois, key=lambda x: x.shape[1]).shape[1]
    #     # print("max width %d" % max_width)
    #     batch_imgs = []
    #     for roi_img in img_rois:
    #         if roi_img.shape[0] < max_width:
    #             new_img = np.ones((roi_img.shape[0], max_width, 1), np.float32)
    #             new_img[:roi_img.shape[0], :roi_img.shape[1], :] = roi_img
    #             batch_imgs.append(new_img)
    #         else:
    #             batch_imgs.append(roi_img)
    #     return batch_imgs

if __name__ == "__main__":
    recoer = Recognizer('./crnn/labels/char_std_5990.txt', \
                        './crnn/models/weights_densenet.h5')
