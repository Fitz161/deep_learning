
"""numpy的切片操作，一般结构如num[a:b,c:d]，分析时以逗号为分隔符，
逗号之前为要取的num行的下标范围(a到b-1)，逗号之后为要取的num列的下标范围(c到d-1)；
前面是行索引，后面是列索引。
如果是这种num[:b,c:d]，a的值未指定，那么a为最小值0；
如果是这种num[a:,c:d]，b的值未指定，那么b为最大值；c、d的情况同理可得。
"""
"""
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpig
 
方式：                        返回类型
OpenCV                        np.ndarray
PIL                           PIL.JpegImagePlugin.JpegImageFile
keras.preprocessing.image     PIL.JpegImagePlugin.JpegImageFile
Skimage.io                    np.ndarray
matplotlib.pyplot             np.ndarray
matplotlib.image              np.ndarray
 
imagePath=""

方式一：使用OpenCV

img1=cv2.imread(imagePath)
print("img1:",img1.shape)
print("img1:",type(img1))
print("-"*10)

方式二：使用PIL

img2=Image.open(imagePath)
print("img2:",img2)
print("img2:",type(img2))
#转换成np.ndarray格式
img2=np.array(img2)
print("img2:",img2.shape)
print("img2:",type(img2))
print("-"*10)
 
方式三：使用keras.preprocessing.image

img3=load_img(imagePath)
print("img3:",img3)
print("img3:",type(img3))
#转换成np.ndarray格式,使用np.array(),或者使用keras里的img_to_array()
#使用np.array()
#img3=np.array(img2)
#使用keras里的img_to_array()
img3=img_to_array(img3)
print("img3:",img3.shape)
print("img3:",type(img3))
print("-"*10)
 
方式四：使用Skimage.io

img4=io.imread(imagePath)
print("img4:",img4.shape)
print("img4:",type(img4))
print("-"*10)

方式五：使用matplotlib.pyplot
'''
img5=plt.imread(imagePath)
print("img5:",img5.shape)
print("img5:",type(img5))
print("-"*10)
 
方式六：使用matplotlib.image

img6=mpig.imread(imagePath)
print("img6:",img6.shape)
print("img6:",type(img6))

"""
import numpy as np
from PIL import Image #安装pillow包

#path = r'D:\photos\bot_cards\cards1\T1_3x.png' #字符串前加r即可不进行转义
path = r'D:\photos\wordcloud\reimu3.jpg'
img = np.array(Image.open(path)) #将图像转成numpy.ndarray
print(img.shape)

#使用之前图像灰度处理中写的一个方法将图像的深度化成1
def flat_img(img:np.ndarray)->np.ndarray:
    gray_img = np.empty(shape = (img.shape[0],img.shape[1]), dtype=np.int32)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            R,G,B = tuple(img[x][y].tolist())
            #ndarray对象使用tolist方法转成list，再通过tuple函数转成三元元组，
            #元组里面元素分别赋值给R G B
            gray_img[x][y] = round(R * 299/1000 + G * 587/1000+ B * 114/1000)
    #之前return位置错了
    return gray_img

def dot_product(array1:np.ndarray, array2:np.ndarray)->int:
    num = 0
    for row in range(array1.shape[0]):
        for col in range(array1.shape[1]):
            num += array1[row][col] * array2[row][col]
    return num

def convolution(image:np.ndarray, kernel_size:tuple=(3,3), stride:int=1)->np.ndarray:
    # np.array是一个函数，用于创建ndarray对象。
    # ndarray是一个类对象，array是一个方法,这里使用的都应是ndarray
    #进行卷积运算后获得的输出图像（矩阵）的高和宽
    output_height = round((image.shape[0] - kernel_size[0]) / stride + 1) #round取整
    output_width = round((image.shape[1] - kernel_size[1]) / stride + 1)
    print(output_height, output_width)
    kernel = np.random.randint(-1, 2, size=kernel_size, dtype=np.int32) #随机生成一个卷积核，randint包左不包右
    #kernel = np.array([[-1,0,-1],[0,4,0],[-1,0,-1]])
    #kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    output_matrix = np.ones((output_height, output_width), dtype=np.int32) #创建输出矩阵
    #np.ones中传shape时高和宽反了
    for row in range(0, output_height): #range同样包左不包右，第三个参数可指定步长
        for col in range(0, output_width):
            output_matrix[row][col] = dot_product(
                image[row*stride : row*stride + kernel_size[0], col*stride : col*stride + kernel_size[1]], kernel)
            #output_matrix[row,col] = kernel.dot(image[row:row + kernel_size[0], col:col + kernel_size[1]])
            #output_matrix[row][col] = np.dot(image[row:row+kernel_size[0], col:col+kernel_size[1]], kernel)
            #这一句一直出现错误：ValueError: setting an array element with a sequence.
            #需将所有ndarray的dtype都统一,发现kenel为int32类型，而image为np.uint8
            #之后发现创建kernel时错误，uint8要为为非负，都改成int32或int64
            #np.dot获得内积，image[a:b, c:d]进行二维数组切片,切片仍是包左不包右
    print(output_matrix.shape)
    return output_matrix

import cv2
cv2.imshow('img', convolution(flat_img(img), (3, 3), 3).astype(dtype=np.uint8))
cv2.waitKey(0)
"""
convolution函数中np.ones函数传入矩阵shape时宽和高写反了
np.dot函数进行的是矩阵乘法，得到的一般为矩阵，而我们需要的是带权和，是一个数
重新编写了一个函数用于计算两个矩阵对应元素乘积的和
最后cv2.show时报错cv2.error: OpenCV(4.4.0) d:\bld\libopencv_1602539518130\work\modules\highgui\src\precomp.hpp:137: 
error: (-215:Assertion failed) src_depth != CV_16F && src_depth != CV_32S in function 'convertToShow'
发现需要将图像转换为标准化为np.uint8（0,255）或np.float32（0,1.0）
"""
