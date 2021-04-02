"""region of intest从图片中提取需要的部分"""
import cv2
import numpy as np

edge_img:np.ndarray = cv2.imread('gray.jpg', cv2.IMREAD_GRAYSCALE)
mask = np.zeros_like(edge_img) #Return an array of zeros with the same shape and type as a given array.

mask = cv2.fillPoly(mask, np.array([[[530,300],[380,300],[160,408],[700,408]]]), 255)
#参数为图像对象，顶点组成的np数组用来指定mask掩码范围，mask颜色，255代表白色
#这样生成了一个掩码，取定的范围为白色（255），其余范围黑色（0）

cv2.imshow('mask', mask)
cv2.waitKey(0)
#最后进行布尔运算，从原图中获取指定的部分 cv2.bitewise_and /np.bitwise_and
masked_edge_img = cv2.bitwise_and(edge_img, mask)

cv2.imshow('img', masked_edge_img)
cv2.waitKey(0)
