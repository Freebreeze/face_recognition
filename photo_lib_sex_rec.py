import cv2
from keras.models import load_model
import numpy as np
# import chineseText

filename =  r'C:\Users\zhao1\Desktop\show\test\beauty\2.jpg'
casvade_face_name='C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_frontalface_default.xml'
casvade_eye = 'C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_eye.xml'
eye_glasses='C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_eye_tree_eyeglasses.xml'


def detect(filename) :
    face_cascade=cv2.CascadeClassifier(casvade_face_name)

    img=cv2.imread(filename)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(140, 140))
    gender_classifier = load_model(
        "C:/Users/zhao1/Desktop/show/test/cascades/simple_CNN.81-0.96.hdf5")
    gender_labels = {0: 'girl', 1: 'boy'}
    color = (255, 255, 255)

    for (x,y,w,h) in faces:
        face = img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, 0)
        face = face / 255.0
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]
        cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        cv2.putText(img, gender, (x, y - 30), font, 1.2, (255, 255, 255), 2)  # 添加文字，1.2表示字体大小，（0,40）是初始的位置，
    cv2.namedWindow('find')
    cv2.imshow('face',img)
    cv2.imwrite('C:/Users/zhao1/Desktop/show/test/人脸.jpg ',img)
    cv2.waitKey(0)
detect(filename)
