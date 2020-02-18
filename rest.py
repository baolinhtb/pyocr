# -*- coding: utf-8 -*-
import io
import os
import time

import cv2
import numpy as np
from flask import Flask, request, redirect, url_for,jsonify,render_template
from flask_cors import CORS

from detector import Detector
detector = Detector('./ctpn/checkpoints','./ctpn/data/text.yml')
from recognizer import Recognizer
recoer = Recognizer('./crnn/labels/char_std_5990.txt', './crnn/models/weights_densenet.h5')

app = Flask(__name__)
app.results = []
app.config['UPLOAD_FOLDER'] = 'samples'
app.config["CACHE_TYPE"] = "null"
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'png', 'jpg', 'jpeg', 'gif'])
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024    # 1 Mb limit
app.config['img'] = app.config['UPLOAD_FOLDER'] + "/idcard.img"
app.filename = ""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def process(fielpath):
    colr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(colr, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    rois, _, img = detector.detect(gray)
    print("CTPN time: %.03fs" % (time.time() - start_time))
    from utilis import sort_box
    rois = sort_box(rois)
    start_time = time.time()
    ocr_result = recoer.recognize(img,rois)
    print("CRNN time: %.03fs" % (time.time() - start_time))

    # sorted_data = sorted(zip(rois, ocr_result), key=lambda x: x[0][1] + x[0][3] / 2)
    # rois, ocr_result = zip(*sorted_data)

    res = {"results": []}

    for i in range(len(rois)):
        res["results"].append({
            'position': rois[i],
            'text': ocr_result[i][1]
        })

    return res

import json

@app.route('/ocr', methods=['POST'])
def ocr():
    if request.method == 'POST':
        img = request.files.get('img')
        img.save(app.config['img'],buffer_size=app.config['MAX_CONTENT_LENGTH'])
        ret = process(app.config['img'])
        return json.dumps(ret)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
