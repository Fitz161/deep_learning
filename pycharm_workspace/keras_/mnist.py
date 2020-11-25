#encoding=utf8
import numpy as np
from keras.datasets import mnist #有很多种数据集
from keras.utils import np_utils #numpy工具包
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

#load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#导入训练集和测试集,x表示数据，y表示标签 download from the internet
print('x_shape:',x_train.shape)
print('y_shape:',y_train.shape)

#(6000，28，28)->(60000, 784)
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
#第二个参数-1表示列数不确定，会自动判断,/255.0是为了归一化
x_test = x_test.reshape(x_test.shape[0], -1)/255.0

y_train = np_utils.to_categorical(y_train,num_classes=10)
#to_categorical函数用来转标签label数据的格式，num_class指定分类种类
y_test = np_utils.to_categorical(y_test,num_classes=10)

#创建模型，输入784个神经元，输出10个神经元
model = Sequential([
    Dense(units=10, input_dim=784,bias_initializer='one',
         activation='softmax')#偏执，激活函数
])
#定义优化器
sgd = SGD(lr=0.2)

model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
                #定义优化器     loss_function 计算准确率 

#训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)
#每个batch32张图片,训练十个周期
#评估模型
loss,accuracy = model.evaluate(x_test, y_test)
print('\ntest loss:',loss)
print('\naccuracy:',accuracy)