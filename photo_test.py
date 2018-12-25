import cv2

filename =  'C:/Users/zhao1/Desktop/show/test/example/1.jpg'
casvade_face_name='C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_frontalface_default.xml'
casvade_eye = 'C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_eye.xml'
eye_glasses='C:/Users/zhao1/Desktop/show/test/cascades/haarcascade_eye_tree_eyeglasses.xml'


def detect(filename) :
    face_cascade=cv2.CascadeClassifier(casvade_face_name)
    eye_cascade=cv2.CascadeClassifier(casvade_eye)
    eye_glass_cascade=cv2.CascadeClassifier(eye_glasses)

    img=cv2.imread(filename)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray,1.2,3)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color = img[y:y + h, x:x + w]
        glass=eye_glass_cascade.detectMultiScale(roi_gray,1.03,5,0,(20,20))
        for (gx, gy, gw, gh) in glass:
            cv2.rectangle(roi_color, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
        # eyes = eye_cascade.detectMultiScale(roi_gray,1.03,5,0,(20,20))
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
        cv2.putText(img, 'man', (x, y - 30), font, 1.2, (255, 255, 255), 2)  # 添加文字，1.2表示字体大小，（0,40）是初始的位置，
    cv2.namedWindow('find')
    cv2.imshow('face',img)
    cv2.imwrite('C:/Users/zhao1/Desktop/show/test/人脸.jpg ',img)
    cv2.waitKey(0)
detect(filename)
