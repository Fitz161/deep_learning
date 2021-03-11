import cv2
import numpy as np

# img = np.zeros((200, 400))
# cv2.line(img, (10, 10), (200, 100), 255, 3)
# cv2.line(img, (30, 50), (350, 10), 255, 2)
#
# cv2.imshow('', img)
# cv2.imwrite('line.jpg', img)
# cv2.waitKey(0)


# img = cv2.imread('line.jpg', cv2.IMREAD_GRAYSCALE)

edge_img:np.ndarray = cv2.imread('gray.jpg', cv2.IMREAD_GRAYSCALE)
mask = np.zeros_like(edge_img) #Return an array of zeros with the same shape and type as a given array.
mask = cv2.fillPoly(mask, np.array([[[530,300],[380,300],[160,408],[700,408]]]), 255)
#参数为图像对象，顶点组成的np数组用来指定mask掩码范围，mask颜色，255代表白色
#这样生成了一个掩码，取定的范围为白色（255），其余范围黑色（0）

#最后进行布尔运算，从原图中获取指定的部分 cv2.bitewise_and /np.bitwise_and
img = cv2.bitwise_and(edge_img, mask)
cv2.imshow('', img)
cv2.waitKey(0)

#霍夫变换只能用于灰度图
lines = cv2.HoughLinesP(img, 1, np.pi / 180, 15, minLineLength=10, maxLineGap=20)

def calculate_slope(line):
    """calculate slope of a line like object
    :parameter line: np.array([[x_1, y_1, x_2, y_2]])"""
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)

left_lines = [line for line in lines if calculate_slope(line) > 0]
right_lines = [line for line in lines if calculate_slope(line) < 0]
print(left_lines, right_lines, sep='\n')
print(len(lines), len(left_lines), len(right_lines))

def reject_abnormal_lines(lines, threshold):
    """"reject abnormal lines by its slope
    :parameter lines: [np.array([[x_1, y_1, x_2, y_2]]), ...]
    :parameter threshold: the threshold of the difference between mean of slopes and the slope of specified line.
    if the difference is beyond the threshold, the line will be rejected.
    """
    slopes = [calculate_slope(line) for line in lines]
    # while len(lines) > 0 :
    #     mean = np.mean(slopes)
    #     diff = [abs(s - mean) for s in slopes]
    #     #index:int = np.argmax(diff) # Returns the index of the maximum values along an iterable.
    #     index = diff.index(max(diff))
    #     if diff[index] > threshold:
    #         slopes.pop(index)
    #         lines.pop(index)
    #     else:
    #         continue
    for i in range(len(lines)):
        mean = np.mean(slopes)
        diff = [abs(s - mean) for s in slopes]
        #index:int = np.argmax(diff) # Returns the index of the maximum values along an iterable.
        index = diff.index(max(diff))
        if diff[index] > threshold:
            slopes.pop(index)
            lines.pop(index)
        else:
            continue
    return lines

print('before filter:\n', 'left:', len(left_lines), 'right:', len(right_lines))
left_lines = reject_abnormal_lines(left_lines, 0.2)
right_lines = reject_abnormal_lines(right_lines, 0.2)
print('after filter:\n', 'left:', len(left_lines), 'right:', len(right_lines))

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

#print('left:', least_squares_fit(left_lines))
#print('right:', least_squares_fit(right_lines))
"""
left: [[460 293]
       [639 415]]
right:[[249 410]
       [413 298]]
"""
left_line = least_squares_fit(left_lines)
right_line = least_squares_fit(right_lines)

img = cv2.imread(r'D:\photos\snippaste\5\Snipaste_2021-02-02_11-56-06.png', cv2.IMREAD_UNCHANGED)

#注意线段坐标必须为tuple类型
cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255), thickness=5)
cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), color=(0, 255, 255), thickness=5)

cv2.imshow('', img)
cv2.waitKey(0)

# capture = cv2.VideoCapture('video.mp4') #传入数字时（从0开始）表示读取第-个摄像头
# ret, frame = capture.read() #获取一帧，两个返回值：视频状况（是否关闭），当前帧的图像np.ndarray
while False:
    capture = cv2.VideoCapture('video.mp4')
    ret, frame = capture.read()
    cv2.imshow('', frame)
    cv2.waitKey(1000 / 24)
