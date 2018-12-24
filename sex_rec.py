# -*- coding:utf-8 -*-
'''
内容：训练图片处理和人脸识别的训练部分
'''
from PIL import Image
import sys
import dlib
import cv2
import os
import os.path
import numpy as np
import PIL.Image
from pylab import *

'''
   利用别人写好的人脸分类器来截取图片中的人脸，并保存到spath中
   i:截取图片保存名
   path:原图片路径
   spath：处理过后图片路径
'''

def get_face_from_photo(i,path,spath):

    detector = dlib.get_frontal_face_detector() #获取人脸分类
    # 读取path路径下的图片，获得所有的图片名字
    filenames = os.listdir(path)

    for f1 in filenames:
        f = os.path.join(path,f1)
        iimag = PIL.Image.open(f)
        # opencv 读取图片，并显示
        img = cv2.imread(f, cv2.IMREAD_COLOR)

        b, g, r = cv2.split(img)    # 分离三个颜色通道
        img2 = cv2.merge([r, g, b])   # 生成新图片

        counts = detector(img, 1) #人脸检测 

        for index, face in enumerate(counts):

            # 在图片中标注人脸，并显示
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()

            #保存人脸区域
            j =str(i)
            j = j+'.jpg'
            save_path = os.path.join(spath,j)
            region = (left,top,right,bottom)
            #裁切图片
            cropImg = iimag.crop(region)

            #保存裁切后的图片
            cropImg.save(save_path)
            i +=1

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(f, img)

    # 等待按键，退出，销毁窗口
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return i

'''
    将人脸图片转化为28*28的灰度图片
    path:人脸图片
    spath:灰度图片
'''
def change_photo_size28(path,spath):


    filenames = os.listdir(path)

    for filename in filenames:
        f = os.path.join(path,filename)
        iimag = PIL.Image.open(f).convert('L').resize((28,28))
        savepath = os.path.join(spath,filename)
        #savepath = spath + '/' + filename
        iimag.save(savepath)

'''
    读取训练图片
'''
def read_photo_for_train(k,photo_path):

    for i in range(k):
        j = i
        j = str(j)
        st = '.jpg'
        j = j+st
        j = os.path.join(photo_path,j)
        im1 = array(Image.open(j).convert('L'))
        #（28，28）-->(28*28,1)
        im1 = im1.reshape((784,1))
        #把所有的图片灰度值放到一个矩阵中
        #一列代表一张图片的信息
        if i == 0:
            im = im1
        else:
            im = np.hstack((im,im1))
    return im


def layerout(w,b,x):

    '''
    sigmoid函数实现
    '''

    y = np.dot(w,x) + b
    t = -1.0*y
    # n = len(y)
    # for i in range(n):
        # y[i]=1.0/(1+exp(-y[i]))
    y = 1.0/(1+exp(t))
    return y

'''
    训练样本：中国某些明星的google图片(106张，女60张，男46张），侵删。女生标签为0，男生标签为1.
    训练方法：简单的梯度下降法
    参考（本人博客另一篇）：https://blog.csdn.net/yunyunyx/article/details/80539222
'''

'''
    设置一个隐藏层，784-->隐藏层神经元个数-->1
'''
def mytrain(x_train,y_train):


    # step=int(input('mytrain迭代步数：'))
    # a=double(input('学习因子：'))
    step=300
    a=0.28
    inn = 784  #输入神经元个数
    # hid = int(input('隐藏层神经元个数：'))#隐藏层神经元个数
    hid=28
    out = 1  #输出层神经元个数

    w = np.random.randn(out,hid)
    w = np.mat(w)
    b = np.mat(np.random.randn(out,1))
    w_h = np.random.randn(hid,inn)
    w_h = np.mat(w_h)
    b_h = np.mat(np.random.randn(hid,1))

    for i in range(step):
        #打乱训练样本
        r=np.random.permutation(104)
        x_train = x_train[:,r]
        y_train = y_train[:,r]
        #mini_batch
        for j in range(100):
            x = np.mat(x_train[:,j])
            x = x.reshape((784,1))
            y = np.mat(y_train[:,j])
            y = y.reshape((1,1))
            hid_put = layerout(w_h,b_h,x)
            out_put = layerout(w,b,hid_put)

            #更新公式的实现
            o_update = np.multiply(np.multiply((y-out_put),out_put),(1-out_put))
            h_update = np.multiply(np.multiply(np.dot((w.T),np.mat(o_update)),hid_put),(1-hid_put))

            outw_update = a*np.dot(o_update,(hid_put.T))
            outb_update = a*o_update
            hidw_update = a*np.dot(h_update,(x.T))
            hidb_update = a*h_update

            w = w + outw_update
            b = b+ outb_update
            w_h = w_h +hidw_update
            b_h =b_h +hidb_update

    return w,b,w_h,b_h
'''
    预测结果pre大于0.5，为男；预测结果小于或等于0.5为女
'''
def mytest(x_test,w,b,w_h,b_h):

    hid = layerout(w_h,b_h,x_test);
    pre = layerout(w,b,hid);
    print(pre)
    if pre > 0.5:
        print("hello,boy!")
    else:
        print("hello,girl!")



#训练

#框出人脸，并保存到faces中,i为保存的名字
i = 0
#女孩
path = 'C:\\Users\\zhao1\\Desktop\\show\\test\\beauty'
spath = 'C:\\Users\\zhao1\\Desktop\\show\\test\\faces'
i = get_face_from_photo(i,path,spath)
# 男孩
path = 'C:\\Users\\zhao1\\Desktop\\show\\test\\nan'
i = get_face_from_photo(i,path,spath)

#将人脸图片转化为28*28的灰度图片
path = 'C:\\Users\\zhao1\\Desktop\\show\\test\\faces'
spath = 'C:\\Users\\zhao1\\Desktop\\show\\test\\grayfaces'
change_photo_size28(path,spath)


#获取图片信息
im = read_photo_for_train(104,spath)

#归一化
immin = im.min()
immax = im.max()
im = (im-immin)/(immax-immin)

x_train = im

#制作标签，前54张是女生，为0
y1 = np.zeros((1,54))
y2 = np.ones((1,50))
y_train = np.hstack((y1,y2))

#开始训练
print("----------------------开始训练-----------------------------------------")
w,b,w_h,b_h = mytrain(x_train,y_train)
print("-----------------------训练结束------------------------------------------")

# 测试
# print("--------------------视频测试-----------------------------------------")
#
# face_detector = dlib.get_frontal_face_detector() #获取人脸分类
# while(True):
#     ret, frame=camera.read()

# path = 'C:\\Users\\zhao1\\Desktop\\show\\test\\girltest'
# spath = 'C:\\Users\\zhao1\\Desktop\\show\\test\\girltest-face'
# i = get_face_from_photo(i,path,spath)
#
# #将人脸图片转化为28*28的灰度图片
# path = 'C:\\Users\\zhao1\\Desktop\\show\\test\\girltest-face'
# spath = 'C:\\Users\\zhao1\\Desktop\\show\\test\\girltest-grayface'
# change_photo_size28(path,spath)
#
#
# #获取图片信息
# im = read_photo_for_train(6,spath)
#
# #归一化
# immin = im.min()
# immax = im.max()
# im = (im-immin)/(immax-immin)
#
# x_test = im
# #print(x_test.shape)
# for i in range(6):
#     xx = x_test[:,i]
#     xx = xx.reshape((784,1))
#     mytest(xx,w,b,w_h,b_h)


# 测试
print("--------------------测试女生-----------------------------------------")
#框出人脸，并保存到girltests中,i为保存的名字
i = 0
#女孩测试集
path = 'C:\\Users\\zhao1\\Desktop\\show\\test\\girltest'
spath = 'C:\\Users\\zhao1\\Desktop\\show\\test\\girltest-face'
i = get_face_from_photo(i,path,spath)

#将人脸图片转化为28*28的灰度图片
path = 'C:\\Users\\zhao1\\Desktop\\show\\test\\girltest-face'
spath = 'C:\\Users\\zhao1\\Desktop\\show\\test\\girltest-grayface'
change_photo_size28(path,spath)


#获取图片信息
im = read_photo_for_train(6,spath)

#归一化
immin = im.min()
immax = im.max()
im = (im-immin)/(immax-immin)

x_test = im
#print(x_test.shape)
for i in range(6):
    xx = x_test[:,i]
    xx = xx.reshape((784,1))
    mytest(xx,w,b,w_h,b_h)

# print("---------------------测试男生-----------------------------")
# #框出人脸，并保存到boytests中,i为保存的名字
# i = 0
# #男孩测试集
# path = 'C:\\Users\\yxg\\Desktop\\boytest'
# spath = 'C:\\Users\\yxg\\Desktop\\boytests'
# i = get_face_from_photo(i,path,spath)
#
# #将人脸图片转化为28*28的灰度图片
# path = 'C:\\Users\\yxg\\Desktop\\boytests'
# spath = 'C:\\Users\\yxg\\Desktop\\boytests'
# change_photo_size28(path,spath)
#
#
# #获取图片信息
# im = read_photo_for_train(6,spath)
#
# #归一化
# immin = im.min()
# immax = im.max()
# im = (im-immin)/(immax-immin)
#
# x_test = im
# for i in range(6):
#     xx = x_test[:,i]
#     xx = xx.reshape((784,1))
#     mytest(xx,w,b,w_h,b_h)