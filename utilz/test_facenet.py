
from pyseeta import Detector
from pyseeta import Aligner
from pyseeta import Identifier

try:
    import cv2
    import numpy as np
except ImportError:
    raise ImportError('opencv can not be found!')

DET_MODEL_PATH = "cfnn/models/seeta_fd_frontal_v1.0.bin"

def test_detector(filepath):
    print('test detector:')
    # load model
    detector = Detector(DET_MODEL_PATH)
    detector.set_min_face_size(30)

    image_color = cv2.imread(filepath, cv2.IMREAD_COLOR)
    h,w,c = image_color.shape
    size = 400.0
    resized = cv2.resize(image_color,(int(w*size/h),int(size)))
    image_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    faces = detector.detect(image_gray)

    for i, face in enumerate(faces):
        print('({0},{1},{2},{3}) score={4}'.format(face.left, face.top, face.right, face.bottom, face.score))
        cv2.rectangle(resized, (face.left, face.top), (face.right, face.bottom), (0,255,0), thickness=2)
        cv2.putText(resized, str(i), (face.left, face.bottom),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), thickness=1)
    cv2.imshow('test', resized)
    cv2.waitKey(0)

    detector.release()

if __name__ == '__main__':
    test_detector('samples/extra/10000031_front.jpg')
