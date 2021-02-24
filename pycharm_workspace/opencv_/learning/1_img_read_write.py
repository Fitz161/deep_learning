import cv2

"""
cv2.imread(‘图像名称’，’可选参数’)
可选参数决定读入图像的模式：
cv2.IMREAD_GRAYSCALE / 0：读入的为灰度图像（即使图像为彩色的）
cv2.IMREAD_COLOR / 1：读入的图像为彩色的（默认）；
cv2.IMAGE_GRAYSCALE 会将图片读成灰度图，只有一个通道
cv2.IMAGE_COLOR 将图片读成三个通道，无论原来几个通道
cv2.IMAGE_UNCHANGED / -1 保持原来图片通道数 
注意的是：即使图像在工作空间不存在，这个函数也不会报错，只不过读入的结果为none
OpenCV目前支持读取bmp、jpg、png、tiff等常用格式
"""
img = cv2.imread(r'D:\photos\bot_cards\cards1\T1_3x.png', cv2.IMREAD_UNCHANGED)
cv2.namedWindow('image') #create a new window
print(img.shape)
cv2.imshow('image', img)
cv2.waitKey(0)

"""使用库包matplotlib来显示图像
from matplotlib import pyplot as plt

plt.imshow(img, 'gray') #必须规定为显示的为什么图像
plt.xticks([]),plt.yticks([]) #隐藏坐标线
plt.show()
"""

"""
新的OpenCV的接口中没有CreateImage接口。即没有cv2.CreateImage这样的函数。如果要创建图像，需要使用numpy的函数
"""
import numpy as np
emptyImage = np.zeros(img.shape, np.uint8)
emptyImage2 = np.zeros_like(img)

emptyImage3 = img.copy() # 复制原有的图像来获得一副新图像

emptyImage4 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #用cvtColor获得原图像的副本。
#cvtColor  convert转换， cv2.COLOR_BGR2GRAY  将RGB图像转成gray图

emptyImage5 = np.zeros(img.shape, dtype=img.dtype)

emptyImage5[:,:,:] = img[:,:,:] if img.ndim == 3 else img[:,:]

cv2.imshow('zeros', emptyImage)
cv2.imshow('zeros_like', emptyImage2)
cv2.imshow('copy', emptyImage3)
cv2.imshow('cvtColor', emptyImage4)
cv2.imshow('index', emptyImage5)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('1.jpg', emptyImage3, [cv2.IMWRITE_JPEG_QUALITY, 50])
"""
第三个参数针对特定的格式： 对于JPEG，其表示的是图像的质量，用0-100的整数表示，默认为95
对于png ,第三个参数表示的是压缩级别0-9。默认为3.
[cv2.IMWRITE_JPEG_QUALITY, 95]
[cv2.IMWRITE_PNG_COMPRESSION, 9]
"""
