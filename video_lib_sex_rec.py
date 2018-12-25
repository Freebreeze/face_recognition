import cv2
from keras.models import load_model
import numpy as np

def detect():

#filename =  'C:/Users/zhao1/Desktop/show/test/example/1.jpg'
    casvade_face='C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_frontalface_default.xml'
    # casvade_eye='C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_eye.xml'
    # eye_glasses='C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_eye_tree_eyeglasses.xml'

    face_cascade=cv2.CascadeClassifier(casvade_face)
    # eye_cascade=cv2.CascadeClassifier(casvade_eye)
    # eye_glass_cascade=cv2.CascadeClassifier(eye_glasses)
    gender_classifier = load_model(
        "C:/Users/zhao1/Desktop/show/test/cascades/simple_CNN.81-0.96.hdf5")
    gender_labels = {0: 'girl', 1: 'boy'}
    color = (255,0,0)
    camera= cv2.VideoCapture(0)

    while(True):
        ret, frame=camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5,0,(40,40))

        for (x,y,w,h) in faces:
            face = frame[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, 0)
            face = face / 255.0
            gender_label_arg = np.argmax(gender_classifier.predict(face))
            gender = gender_labels[gender_label_arg]
            cv2.rectangle(frame, (x, y), (x + h, y + w), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
            cv2.putText(frame, gender, (x, y - 30), font, 1.2, (255, 255, 255), 2)  # 添加文字，1.2表示字
            cv2.imshow("camera",frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()
if __name__ =="__main__":
    detect()

