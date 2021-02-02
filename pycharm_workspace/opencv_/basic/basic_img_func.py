import cv2

"""
cv2.IMAGE_GRAYSCALE 会将图片读成灰度图，只有一个通道
cv2.IMAGE_COLOR 将图片读成三个通道，无论原来几个通道
cv2.IMAGE_UNCHANGED 保持原来图片通道数 
"""
img = cv2.imread(r'D:\photos\bot_cards\cards1\T1_3x.png', cv2.IMREAD_GRAYSCALE)
print(type(img))
print(img.shape)

cv2.imshow('img', img)
#key = cv2.waitKey(2000) #单位毫秒,参数设为0则可一直等待,返回输入键盘字符的ASCII值
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
else:
    cv2.imshow('img', img)
    cv2.waitKey(2000)

cv2.imwrite('1.jpg', img)#保存为jpg会压缩，bmp无损
