
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
# 保存模型，权重
import tempfile
from keras.models import save_model, load_model
import numpy as np

def load_data(path='mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
# 将数据读入
(x_train, y_train), (x_test, y_test) = load_data()

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# plot 4 images as gray scale
# plt.subplot(221)
# print(y_train[4545],y_train[1],y_train[2],y_train[3])
# plt.imshow(x_train[4545], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
# # show the plot
# plt.show()
X_train = x_train.reshape(x_train.shape[0], 784)
X_test = x_test.reshape(x_test.shape[0],784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# print(X_train.shape,X_test.shape)
# print(X_train.dtype,X_test.dtype)
# 归一化
X_train /= 255
X_test /= 255

#  对于y 独热编码
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# 初始化序列模型：
model = Sequential()
#model.add(Dense(10,input_dim=784))
#model.add(Activation('softmax'))
model.add(Dense(128,input_dim=784,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
#model.summary()
model.compile(optimizer=keras.optimizers.sgd(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, Y_train, nb_epoch=20, batch_size=128,verbose=1,validation_split=0.2)

model.save('mnist_kerasModel.h5')
#测试模型
print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, Y_test,verbose=1)
print('test loss: ', loss)
print('test accuracy: ', accuracy)