import cv2
import numpy as np

"""访问像素"""
img = cv2.imread(r'D:\photos\bot_cards\cards1\T1_3x.png', cv2.IMREAD_UNCHANGED)
print(img[0, 0, 0])  # numpy数组使用下标访问其中元素，各维度索引用逗号隔开
# img[i, j, k] 即取第k个通道 (获得二维图像矩阵) 的第i行第j列的那个元素的数值
print(img.ndim)  # 获取img对象对应的矩阵的维度，灰度图一般为2维， 其余3维
print(img.shape) #四通道图像RGBA，有alpha通道

"""
与C++不同，在Python中灰度图的img.ndim = 2，而C++中灰度图图像的通道数img.channel() =1
这里使用了numpy的随机数，Python自身也有一个随机数生成函数。这里只是一种习惯，np.random模块中拥有更多的方法，
而Python自带的random只是一个轻量级的模块。不过需要注意的是np.random.seed()不是线程安全的，
而Python自带的random.seed()是线程安全的。如果使用随机数时需要用到多线程，建议使用Python自带的random()和random.seed()，
或者构建一个本地的np.random.Random类的实例
"""
def salt(img, n):  # 加盐
    for k in range(n):
        # 随机获取图像内的像素点的坐标
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        #将该像素设为255（灰度图蛋通道255即为白色，rgb需三个通道都设为白色）
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
salt(img, 100)
cv2.imshow('', img)
cv2.waitKey(0)

"""分离、合并通道"""
img = cv2.imread('1.jpg', 1)

b, g, r = cv2.split(img) #分离有多个通道的图像，返回各个通道组成的元组
"""注意顺序是BGR"""

g = np.zeros( (img.shape[0], img.shape[1]), dtype=img.dtype)
g[:,:] = img[:,:,1]
#冒号表示选取该维度所有的数据，逗号分隔不同维度的索引

cv2.imshow("Blue", r)
cv2.imshow("Red", g)
cv2.imshow("Green", cv2.split(img)[2]) #用下标来取第三个通道

cv2.waitKey(0)
cv2.destroyAllWindows()

"""通道合并"""
merged = cv2.merge([b, g, r])
mergedByNp = np.dstack([b, g, r])

# NumPy数组的strides属性表示的是在每个维数上以字节计算的步长,即在每个维度上以元素（可能是个ndarray）计算该元素占的字节数
print(merged.strides, mergedByNp.strides) #(1800, 3, 1) (1800, 3, 1)
cv2.imshow("merged", merged)
cv2.imshow("mergedByNp", mergedByNp)

cv2.waitKey(0)
cv2.destroyAllWindows()

arr = np.arange(0, 27).reshape((3, 3, 3))
print(arr.strides) #(36, 12, 4) <== 3×3×4，3×4，4


