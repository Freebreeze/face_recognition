import numpy as np
import cv2
filename =  'C:/Users/zhao1/Desktop/show/test/example/1.jpg'
casvade_face_name='C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_frontalface_default.xml'
casvade_eye = 'C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_eye.xml'


# 脸
face_cascade = cv2.CascadeClassifier(casvade_face_name)
# 眼睛
eye_cascade = cv2.CascadeClassifier(casvade_eye)
# # 嘴巴
# mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
# # 鼻子
# nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
# # 耳朵
# leftear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
# rightear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

# face_cascade = cv2.CascadeClassifier("../../opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier('../../opencv-2.4.9/data/haarcascades/haarcascade_eye.xml')

img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 脸
faces = face_cascade.detectMultiScale(gray, 1.2, 3)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    # 眼睛
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 3)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    # # 嘴巴
    # mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 5)
    # for (mx, my, mw, mh) in mouth:
    #     cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
    # # 鼻子
    # nose = nose_cascade.detectMultiScale(roi_gray, 1.2, 5)
    # for (nx, ny, nw, nh) in nose:
    #     cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)
    #
    # # 耳朵
    # leftear = leftear_cascade.detectMultiScale(roi_gray, 1.01, 2)
    # for (lx, ly, lw, lh) in leftear:
    #     cv2.rectangle(roi_color, (lx, ly), (lx + lw, ly + lh), (0, 0, 0), 2)
    #
    # rightear = rightear_cascade.detectMultiScale(roi_gray, 1.01, 2)
    # for (rx, ry, rw, rh) in rightear:
    #     cv2.rectangle(roi_color, (rx, ry), (rx + rw, ry + rh), (0, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

