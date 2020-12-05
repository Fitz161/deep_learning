import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Convolution2D #二维卷积
from keras.layers import MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

#载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#(60000，28，28)->(60000, 28, 28, 1) #深度为1
x_train = x_train.reshape(-1, 28, 28, 1)/255.0
#参数-1表示列数不确定，会自动判断,/255.0是为了归一化，即变为0-1间的数
x_test = x_test.reshape(-1, 28, 28, 1)/255.0

#换one hot模式
y_train = np_utils.to_categorical(y_train,num_classes=10)
#to_categorical函数用来转标签label数据的格式，num_class指定分类种类
y_test = np_utils.to_categorical(y_test,num_classes=10)

#定义顺序模型
model = Sequential()

#定义第一个卷积层
model.add(Convolution2D(
    input_shape = (28, 28, 1), #输入平面
    filters = 32, #卷积核/滤波器个数
    kernel_size = 5, #卷积窗口大小
    strides = 1, #步长
    padding = 'same', #padding方式： same/valid
    activation = 'relu')) #激活函数

#第一个化池层，变成14*14
model.add(MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same'))
#第二个卷积层 14*14,64个特征图
model.add(Convolution2D(
    filters = 64,
    kernel_size = 5,
    strides = 1, #步长
    padding = 'same', #padding方式： same/valid
    #same表示经过卷积之后图仍然是28*28的不改变大小
    activation = 'relu')) #激活函数
#第二个化池层,变成7*7， 64个特征图
model.add(MaxPooling2D(
    pool_size = 2,
    strides = 2,
    padding = 'same'))
#把第二个化池层的输出扁平化为一维,64*7*7的长度
model.add(Flatten())
#第一个全连接层
model.add(Dense(1024, activation='relu'))
#Dropout
model.add(Dropout(0.5)) #训练时一半神经元工作
#第二个全连接层
model.add(Dense(10, activation='softmax'))

#定义优化器
adam = Adam(lr=1e-4)
#定义优化器 loss_function 计算准确率 
model.compile(optimizer=adam, loss='categorical_crossentropy',
             metrics=['accuracy'])

#训练模型 每个batch64张图片,训练十个周期
model.fit(x_train, y_train, batch_size=64, epochs=10)
#评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss:',loss)
print('\naccuracy:',accuracy)