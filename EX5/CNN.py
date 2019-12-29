import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import keras
import h5py
from keras.models import Sequential
from keras.layers import InputLayer,Input,MaxPooling2D,Conv2D,Dense,Flatten,Reshape,Convolution2D
from keras.optimizers import adam
from keras.utils import to_categorical
from keras.models import Model
# 保存模型，权重
import tempfile
from keras.models import save_model, load_model


def load_data(path='../EX4/mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
# 将数据读入
(x_train, y_train), (x_test, y_test) = load_data()
#
data = x_train[0]
img_size = data.shape[0]
img_size_flat = data.shape[0] * data.shape[1]
img_shape = (data.shape[0],data.shape[1])
img_shape_full = (28,28,1)
num_classes = 10
num_channels = 1


# print(x_train[0].shape)
# print(y_train[0])
# print(img_shape)
# print(img_shape_full)
# print(num_classes)
# print(num_channels)


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # 创建3x3子图.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # 画图.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # 显示真正的预测的类别.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # 将类别作为x轴的标签
        ax.set_xlabel(xlabel)

        # 去除图中的刻度线.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

# plot_images(x_train[0:9],y_train[0:9])
# plt.show()


def plot_example_errors(cls_pred, correct,x_text,y_text):
    # 此函数从下面的print_test_accuracy()调用
    # cls_pred包含了测试集所有图像的预测类别
    # correct是布尔数组，表示预测的类是否等于测试集中每个图像的真实类别.
    # 布尔数组中未正确分类的图像
    incorrect = (correct == False)

    # 从测试集中获取未正确分类的图片
    images = x_text[incorrect]

    # 获取这些图像的预测列表
    cls_pred = cls_pred[incorrect]

    # 获取这些图像的正确类别
    cls_true = y_text[incorrect]

    # 画出前9张图像
    plot_images(images=images[0:9],cls_true=cls_true[0:9], cls_pred = cls_pred[0:9])



def trainModel(x_train,y_train):
    x_train = x_train.reshape(60000,784)
    y_train = to_categorical(y_train,10)
    # 开始构建Keras 序列模型。
    model = Sequential()
    # 添加一个输入层，类似于TensorFlow中的feed_dict。
    # 输入形状input-shape 必须是包含图像大小image-size[从代码中看起来应该是img_size_flat？]_flat的元组。
    model.add(InputLayer(input_shape=(784,)))#如果使用老师的代码也就是有 Inputlayr层那么模型就不能进行正常的存储，还有读取。

    # 输入是一个包含784个元素的扁平数组，
    # 但卷积层期望图像形状是（28,28,1）
    model.add(Reshape(img_shape_full))
    model.add(Convolution2D(25, (5, 5), input_shape=(28, 28, 1)))
    # 具有ReLU激活和最大池化的第一个卷积层。
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation='relu', name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # 具有ReLU激活和最大池化的第二个卷积层
    model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation='relu', name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # 将卷积层的4级输出展平为2级，可以输入到完全连接/密集层。
    model.add(Flatten())

    # 具有ReLU激活的第一个完全连接/密集层。
    model.add(Dense(128, activation='relu'))

    # 最后一个完全连接/密集层，具有softmax激活功能，用于分类
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=1, validation_split=0.1)

    # X_test = x_test.reshape((10000, 28, 28, 1))
    # Y_test = to_categorical(y_test,10)
    #
    # loss, acc = model.evaluate(X_test, Y_test, verbose=1)
    # print("loss:", loss)
    # print("accuracy:", acc)
    return model

def KerasAPI(x_train,y_train):
    # 初始化数据，
    x_train = x_train.reshape(60000,784)
    y_train = to_categorical(y_train,10)
    # 创建一个输入层，类似于TensorFlow中的feed_dict。.
    # 输入形状input-shape 必须是包含图像大小image_size_flat的元组。.
    inputs = Input(shape=(img_size_flat,))

    # 用于构建神经网络的变量。
    net = inputs

    # 输入是一个包含784个元素的扁平数组
    #  但卷积层期望图像形状是（28,28,1）
    net = Reshape(img_shape_full)(net)
    #  具有ReLU激活和最大池化的第一个卷积层。
    net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)

    #  具有ReLU激活和最大池化的第二个卷积层.
    net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)

    # 将卷积层的4级输出展平为2级，可以输入到完全连接/密集层。
    net = Flatten()(net)

    # 具有ReLU激活的第一个完全连接/密集层。
    net = Dense(128, activation='relu')(net)

    #  最后一个完全连接/密集层，具有softmax激活功能，用于分类
    net = Dense(num_classes, activation='softmax')(net)

    # 神经网络输出
    outputs = net

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=1)

    X_test = x_test.reshape(10000, 784)
    Y_test = to_categorical(y_test,10)
    loss, acc = model.evaluate(X_test, Y_test, verbose=1)
    print("loss:", loss)
    print("accuracy:", acc)
    return model

#输入参数weights是待可视化的权重，input_channel是输入通道的数量，其缺省值为0。
def plot_conv_weights(weights, input_channel=0):
    #获取权重的最低值和最高值
    # 这用于校正图像的颜色强度，以便可以相互比较.
    w_min = np.min(weights)
    w_max = np.max(weights)
    # 卷积层中的卷积核数量
    num_filters = weights.shape[3]
    #要绘制的网格数.
    # 卷积核的平方根.
    num_grids = math.ceil(math.sqrt(num_filters))
    #创建带有网格子图的图像.
    fig, axes = plt.subplots(num_grids, num_grids)
    #  绘制所有卷积核的权重
    for i, ax in enumerate(axes.flat):
        #  仅绘制有限的卷积核权重
        if i<num_filters:
            #  获取输入通道的第i个卷积核的权重
            #   有关于４维张量格式的详细信息请参阅new_conv_layer()
            img = weights[:, :, input_channel, i]
            # 画图
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        # 去除刻度线.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_conv_output(values):
    # 卷积层中的卷积核数量
    num_filters = values.shape[3]
    # 要绘制的网格数
    # 卷积核的平方根
    num_grids = math.ceil(math.sqrt(num_filters))
    # 创建带有网格子图的图像
    fig, axes = plt.subplots(num_grids, num_grids)
    #画出所有卷积核的输出图像
    for i, ax in enumerate(axes.flat):
        # 仅画出有效卷积核图像
        if i<num_filters:
            # 获取第i个卷积核的输出图像
            img = values[0, :, :, i]
            # 画图e.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        # [移除？]移除刻度线
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')
    plt.show()


image1 = x_test[0]
#plot_image(image1)

#model = trainModel(x_train,y_train)
#model.save("CNN.h5")
#model = load_model("CNN.h5")

#训练模型2，存储模型。
model2 = KerasAPI(x_train,y_train)
model2.save_weights("modelWeights.h5")
model2.load_weights("modelWeights.h5",by_name=True)
from keras.models import model_from_json
json = model2.to_json()
model3 = model_from_json(json)


# X_test = x_test.reshape((10000, 28, 28, 1))
# Y_test = y_test

# Y_predict = new_model.predict(X_test)
# Y_predict = [np.argmax(y_predict)for y_predict in Y_predict]
# print(Y_predict)
# y_predict = np.array(Y_predict)
# correct = Y_predict == Y_test
# plot_example_errors(y_predict,correct,x_test,y_test)

#提取出每一层，然后可视化分析每一层
model3.summary()
layer_input = model3.layers[0]
layer_conv1 = model3.layers[2]
layer_conv2 = model3.layers[4]

weights_conv1 = layer_conv1.get_weights()[0]
plot_conv_weights(weights=weights_conv1, input_channel=0)

weights_conv2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights=weights_conv2, input_channel=0)


#画正确的图函数
# plt.figure()
# plot_images(X_test[0:9],Y_test[0:9],Y_predict[0:9])
# plt.show()


from keras import backend as K
output_conv1 = K.function(inputs=[layer_input.input],
                          outputs=[layer_conv1.output])

image1 = image1.reshape(1,784)#!!!很关键。
layer_output1 = output_conv1([[image1]])[0]

#print(layer_output1.shape)
#画图
#plot_conv_output(values=layer_output1)
# 卷积层输出方法二
output_conv2 = Model(inputs=layer_input.input,
                     outputs=layer_conv2.output)
layer_output2 = output_conv2.predict(np.array(image1))#去掉中括号
#print(layer_output2.shape)
plot_conv_output(values=layer_output2)

