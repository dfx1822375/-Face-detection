#运行程序，需要下载三个npy文件
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import mtcnn_detect_face
import cv2


def main():
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()  # 开启会议
    # create_mtcnn.py展示了如何使用模型，即在使用时必须先调用detect_facec的creat_mtcnn方法导入网络结构，
    # 此时在创建时又需要写出对应的网络结构然后通过.npy进行数据恢复然后再使用。
    pnet, rnet, onet = mtcnn_detect_face.create_mtcnn(sess, None)  # 通过加载存储的.npy到对应的网络模型中恢复网络中的参数。
    minsize = 20  # 表示最小的人脸尺寸
    threshold = [0.6, 0.7, 0.9]  # 分别为3个网络中人脸得分的阈值
    factor = 0.709  # 金字塔缩放因子
    videoCapture = cv2.VideoCapture(0)  # 加载摄像头
    while videoCapture.isOpened():  # 判断是否打开成功
        sucess, frame = videoCapture.read()  # 按帧读取
        draw = frame
        img = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        onet_bounding_boxes, points = mtcnn_detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                                    factor)  # 人脸检测
        face_number = 1;
        for b in onet_bounding_boxes:  # 把找到的所有人脸框按行赋给b
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255),3)  # 画框
            print("第%d个人脸框左上角坐标：(%d,%d)" % (face_number, int(b[0]), int(b[1])))
            print("第%d个人脸框右下角坐标：(%d,%d)" % (face_number, int(b[2]), int(b[3])))
            print("第%d个人脸的可能性为：%f" % (face_number, float(b[4])))
            face_number += 1

        for p in points.T:  # 把找到的所有人脸特征点按行赋给p。  .T为转置，把10行n列变成n行10列
            for i in range(5):
                cv2.circle(draw, (int(p[i]), int(p[i + 5])), 1, (0, 0, 255), 2)
        cv2.putText(draw, '%f' % float(b[4]), (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        key = cv2.waitKey(delay=1)
        if (key == ord('q')):
            break

        cv2.namedWindow("onet_bounding_boxes", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("onet_bounding_boxes", 400, 500)
        cv2.imshow("onet_bounding_boxes", draw)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
