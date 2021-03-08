import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img,5)
"""
中值滤波
medianBlur函数使用中值滤波器来平滑（模糊）处理一张图片
C++: void medianBlur(InputArray src, OutputArray dst, int ksize)
第一个参数，InputArray类型的src，函数的输入参数，填1、3或者4通道的Mat类型的图像；当ksize为3或者5的时候，图像深度需为CV_8U，CV_16U，或CV_32F其中之一，而对于较大孔径尺寸的图片，它只能是CV_8U。
第二个参数，OutputArray类型的dst，即目标图像，函数的输出参数，需要和源图片有一样的尺寸和类型。我们可以用Mat::Clone，以源图片为模板，来初始化得到如假包换的目标图。
第三个参数，int类型的ksize，孔径的线性尺寸（aperture linear size），注意这个参数必须是大于1的奇数，比如：3，5，7，9 ...
中值滤波是基于排序统计理论的一种能有效抑制噪声的非线性信号处理技术。它也是一种邻域运算，类似于卷积，但是计算的不是加权求和，
而是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替，让周围像素灰度值的差比较大的像素改取与周围的像素值接近的值，
从而可以消除孤立的噪声点。它能减弱或消除傅立叶空间的高频分量，但影响低频分量。因为高频分量对应图像中的区域边缘的灰度值具有较大
较快变化的部分，该滤波可将这些分量滤除，使图像平滑。值滤波技术在衰减噪声的同时能较好的保护图像的边缘。
"""
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #阈值
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
"""
阈值化操作在图像处理中是一种常用的算法，比如图像的二值化就是一种最常见的一种阈值化操作。opencv2和opencv3中提供了
直接阈值化操作cv::threshold()和自适应阈值化操作cv::adaptiveThreshold()两种阈值化操作接口
threshold(src,thresh,maxval, type,dst=None)#输出图像
src：原图像。
thresh：当前阈值。
maxVal：最大阈值，一般为255.
thresholdType:阈值类型 在二值化操作中常用的阈值类型为THRESH_BINARY,THRESH_BINARY_INV

适应阈值化能够根据图像不同区域亮度分布的，改变阈值
支持两种自适应方法，即cv::ADAPTIVE_THRESH_MEAN_C（平均）和cv::ADAPTIVE_THRESH_GAUSSIAN_C（高斯）。
在两种情况下，自适应阈值T(x, y)。通过计算每个像素周围bxb大小像素块的加权均值并减去常量C得到。
其中，b由blockSize给出，大小必须为奇数；如果使用平均的方法，则所有像素周围的权值相同；如果使用高斯的方法，
则（x,y）周围的像素的权值则根据其到中心点的距离通过高斯方程得到。
void cv::adaptiveThreshold(  
    cv::InputArray src, // 输入图像  
    double maxValue, // 向上最大值  
    int adaptiveMethod, // 自适应方法，平均或高斯  
    int thresholdType // 阈值化类型  
    int blockSize, // 块大小  
    double C // 常量  
);
"""
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()