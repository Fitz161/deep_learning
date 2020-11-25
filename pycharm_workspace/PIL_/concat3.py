import os
from PIL import Image
from random import sample, choices

COL = 10
ROW = 10
UNIT_HEIGHT_SIZE = 900
UNIT_WIDTH_SIZE = 600
PATH = "D:/photos/qq_photo/cards2/"
NAME = 'pic5'
REPEAT_SELECT = False
SAVE_QUALITY = 50

def concat_images(image_names, name, path):
    image_files = []
    for index in range(COL*ROW):
        image_files.append(Image.open(path + image_names[index]))
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW))
    for row in range(ROW):
        for col in range(COL):
            target.paste(image_files[COL*row+col], (0 + UNIT_WIDTH_SIZE*col, 0 + UNIT_HEIGHT_SIZE*row))
    target.save(path + name + '.jpg', quality=SAVE_QUALITY)


def get_image_names(path):
    image_names = list(os.walk(path))[0][2]
    selected_images = choices(image_names, k=COL*ROW) if REPEAT_SELECT else sample(image_names, COL*ROW)
    return selected_images


if __name__ == '__main__':
    concat_images(get_image_names(PATH), NAME, PATH)
