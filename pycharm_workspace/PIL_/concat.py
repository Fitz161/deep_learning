import os
from PIL import Image
from random import sample

UNIT_HEIGHT_SIZE = 900#高
UNIT_WIDTH_SIZE = 600

def concat(images, name):
    image_files = []
    for i in range(len(images)):
        image_files.append(Image.open(path + images[i]))
    target = Image.new('RGB', (UNIT_WIDTH_SIZE*5, UNIT_HEIGHT_SIZE*2))   # result is 5*2白色图
    left_x1 = 0 #left_x1表示第一行左上角的x坐标，x2表示第二行
    left_x2 = 0
    right_x1 = UNIT_WIDTH_SIZE
    right_x2 = UNIT_WIDTH_SIZE
    for i in range(len(image_files)):
        if(i%2==0):
            target.paste(image_files[i], (left_x1, 0, right_x1, UNIT_HEIGHT_SIZE)) #后面的四元素元组表示图片粘贴到的位置的左上角和右下角坐标
            left_x1 += UNIT_WIDTH_SIZE #第一行左上角x坐标右移
            right_x1 += UNIT_WIDTH_SIZE #右下角x坐标右移
        else:
            target.paste(image_files[i], (left_x2, UNIT_HEIGHT_SIZE, right_x2, UNIT_HEIGHT_SIZE * 2))
            left_x2 += UNIT_WIDTH_SIZE #第二行左上角右移
            right_x2 += UNIT_WIDTH_SIZE #右下角右移
    quality_value = 100
    target.save(path+name+'.jpg', quality = quality_value)

path = "D:/photos/qq_photo/cards/"
images = [] # 先存储所有的图像的名称
for root, dirs, files in os.walk(path):
    for f in files :
        images.append(f)
seleted_imgs = sample(images, 10) #从images选10个不重复元素的组成个新列表
print(seleted_imgs)
concat(seleted_imgs,"hello1")
