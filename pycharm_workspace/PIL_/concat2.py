import os
from PIL import Image
from random import sample

UNIT_HEIGHT_SIZE = 900
UNIT_WIDTH_SIZE = 600


def concat_images(image_names, name, path):
    image_files = []
    for i in range(len(image_names)):
        image_files.append(Image.open(path + image_names[i]))
    # 创建3000*1800的RGB图，默认为黑色，可在第三个参数指定颜色
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * 5, UNIT_HEIGHT_SIZE * 2))
    x_distance = 0
    for i in range(len(image_files)):
        if(i % 2 == 0):
            # 传入二元元组，即将图复制到的位置的左上角的坐标
            target.paste(image_files[i], (x_distance, 0))
        else:
            target.paste(image_files[i], (x_distance, UNIT_HEIGHT_SIZE))
            x_distance += UNIT_WIDTH_SIZE
    quality_value = 95
    # quality参数： 保存图像的质量，值的范围从1（最差）到95（最佳）。 默认值为75，使用中应尽量避免高于95的值;
    # 100会禁用部分JPEG压缩算法，并导致大文件图像质量几乎没有任何增益。
    # 使用此参数后，图片大小会增加
    target.save(path + name + '.jpg', quality=quality_value)


def get_image_names(path):
    # image_names = []
    # for root, dirs, images in os.walk(path): #返回一个包含一个三元元组的iterator对象，只能用for in或next（ite）获取元素
    #    image_names = images
    image_names = list(os.walk(path))[0][2] #先用list将iterator转成list，再[0]取出里面唯一的元素--一个三元元组，再[2]
    selected_images = sample(image_names, 10)  # 从images选10个不重复元素的组成个新列表
    return selected_images


if __name__ == '__main__':
    path = "D:/photos/qq_photo/cards2/"
    concat_images(get_image_names(path), "pic3", path)
