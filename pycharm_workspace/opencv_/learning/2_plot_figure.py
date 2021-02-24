import cv2
#import numpy as np
#from matplotlib import pyplot as plt
img = cv2.imread(r'D:\photos\bot_cards\cards1\T1_3x.png', cv2.IMREAD_UNCHANGED)

"""
cv2.line（img,Point pt1,Point pt2,color,thickness=1,line_type=8 shift=0）
点的坐标为二元元组
"""
cv2.line(img, (0, 0), (300, 300), color=(255, 255, 255), thickness=5)
#三个通道，这里color使用了rgb，灰度图传一个即可

"""
绘制矩形
矩形的两个点（左上角与右下角）

绘制旋转矩形
不支持cv::RotateRect ，需要
使用cv::line()逐条边绘制
使用cv::drawContours()函数进行绘制
"""
cv2.rectangle(img, (10, 10), (300, 300), (155, 155, 155), 5)

"""
绘制圆形
给定圆心坐标和半径（px）
"""
cv2.circle(img, (300, 300), 100, 255, 5)

cv2.imshow('', img)
cv2.waitKey(0)
