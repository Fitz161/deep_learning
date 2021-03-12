import cv2

img = cv2.imread(r'D:\photos\snippaste\5\Snipaste_2021-02-02_11-56-06.png', cv2.IMREAD_GRAYSCALE)
#edge_img = cv2.Canny(img, 50, 100) #三个参数分别是图像对象，上下边缘（阈值）
edge_img = cv2.Canny(img, 100, 200) #提高阈值
# 判定某点为边缘，像素值变成255，否则为0

cv2.imshow('img', edge_img)
cv2.imwrite('gray.jpg', edge_img)
cv2.waitKey(0)