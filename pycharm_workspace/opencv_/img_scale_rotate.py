import cv2 as cv
import numpy as np


class ImgProcess():
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        print(f"image name:{name}\nimage path:{path}")
        try:
            self.img = cv.imread(path)
        except BaseException:
            print("failed to open the pic")
        else:
            self.height, self.width, depth = self.img.shape
            print("initialization has been done")

    def translate(self):
        """
        平移是物体位置的移动。如果您知道在(x,y)方向上的位移，则将其设为(tx,ty)，
        你可以创建转换矩阵M，如下所示：
        M=[1 0 tx               [[1, 0, tx],
          0 1  ty], 即python列表 [0, 1, ty]]
        您可以将其放入**np.float32**类型的Numpy数组中
        """
        try:
            x = eval(input("please enter x distance"))
            y = eval(input("please enter y distance"))
        except BaseException:
            print("illegal input, using default distance now")
            x, y = 100, 50
        M = np.float32([[1, 0, x],
                        [0, 1, y]])
        dst = cv.warpAffine(self.img, M, (self.width, self.height))
        # **cv.warpAffine**函数的第三个参数是输出图像的大小，其形式应为(width，height)。
        # 记住width =列数，height =行数。
        print("show image now, type q to quit")
        cv.imshow(name, dst)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print("quit translation func")

    def rotate(self):
        """
        图像旋转角度为θ是通过以下形式的变换矩阵实现的：
        M=[cosθ -sinθ
           sinθ cosθ]
        但是OpenCV提供了可缩放的旋转以及可调整的旋转中心，因此您可以在自己喜欢的任何位置旋转。修改后的变换矩阵为
        [[α  β  (1−α)⋅center.x  −β⋅center.y],
        [−β  α  β⋅center.x+(1−α)⋅center.y]]
        其中:α=scale⋅cosθ,β=scale⋅sinθ
        为了找到此转换矩阵，OpenCV提供了一个函数**cv.getRotationMatrix2D**。
        """
        try:
            angle = eval(input("please input the angel to be rotated"))
        except BaseException:
            print("illegal input\nusing default degree(90) now")
            angle = 90
        M = cv.getRotationMatrix2D(
            (self.height - 1 / 2, self.width - 1 / 2), angle, 1)
        dst = cv.warpAffine(self.img, M, (self.height, self.width))
        print("show image now, type q to quit")
        cv.imshow(name, dst)
        cv.waitKey(0)
        cv.destroyAllWindows()
        print("quit rotate func")

    def flip(self):
        """
        cv2.flip(src, flipCode[, dst]) → dst
        src – 输入的图像
        dst – 输出的图像
        flipCode – 翻转模式，flipCode == 0
        垂直翻转（沿X轴翻转），flipCode > 0
        水平翻转（沿Y轴翻转），flipCode < 0
        水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
        """
        # 水平翻转
        try:
            flipCode = eval(
                input(
                    "please input what kind of flip to be done to the pic\n"
                    "0 for vertically flip; 1 for horizontally flip; -1 for ver and hor flip"))
        except BaseException:
            print("illegal input")
        else:
            dst = cv.flip(self.img, flipCode=flipCode)
            print("show image now, type q to quit")
            cv.imshow('img', dst)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print("quit flip func")


if __name__ == '__main__':
    name = input("please enter the name of the pic")
    path = input("please enter the absolute path of the pic")
    id = ImgProcess(name, path)
    print("calling translate func")
    id.translate()
    print("calling rotate func")
    id.rotate()
    print("calling flip func")
    id.flip()
