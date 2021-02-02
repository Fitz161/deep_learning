import cv2
import numpy as np

arr = np.array([[1, 2], [3, 4]])
print(np.ravel(arr))
print(arr.ravel())
#print(list(arr.flat))

poly = np.polyfit([0, 2, 3, 4], [1, 2, 4, 6], deg=1) #[1.22857143 0.48571429]
#多项式拟合函数，上述两数组分别给定x，y坐标值， deg表示拟合的次方
#deg = 1表示拟合为y = kx + b一次直线 ，返回 [k, b]

l = lambda x : poly[0] * x + poly[1]
print(l(100))

print(np.polyval(poly, 100)) #多项式求值，结果同上

def least_squares_fit(lines):
    """
    将lines中的线段拟合成一条线段
    :param lines:线段列表
    :return: np.array([ [x_min, y_in], [x_max, y_max] ])
    """
    x_coordinates = np.ravel([ [line[0][0], line[0][2] ] for line in lines])
    y_coordinates = np.ravel([ [line[0][1], line[0][3] ] for line in lines])

    poly = np.polyfit(x_coordinates, y_coordinates, deg=1)
    point_min = (np.min(x_coordinates), np.polyval(poly, np.min(x_coordinates)))
    point_max = (np.max(x_coordinates), np.polyval(poly, np.max(x_coordinates)))
    return np.array([point_min, point_max], dtype=np.int)


