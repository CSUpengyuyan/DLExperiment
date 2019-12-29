import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor, log
import os

output_dir = "output/"
# 处理数据 x
def dataProcess_X(rawData):

    #sex 只有两个属性 先drop之后处理
    if "income" in rawData.columns:
        Data = rawData.drop(["sex", 'income'], axis=1)
    else:
        Data = rawData.drop(["sex"], axis=1)
    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"] #读取非数字的column
    listNonObjedtColumn = [x for x in list(Data) if x not in listObjectColumn] #数字的column

    ObjectData = Data[listObjectColumn]
    NonObjectData = Data[listNonObjedtColumn]
    #insert set into nonobject data with male = 0 and female = 1
    NonObjectData.insert(0 ,"sex", (rawData["sex"] == " Female").astype(np.int))
    #set every element in object rows as an attribute
    ObjectData = pd.get_dummies(ObjectData)

    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data_x = Data.astype("int64")
    #normalize
    Data_x = (Data_x - Data_x.mean()) / Data_x.std()

    return Data_x
# 处理数据 y
def dataProcess_Y(rawData):
    df_y = rawData['income']
    Data_y = pd.DataFrame((df_y==' >50K').astype("int64"), columns=["income"])
    return Data_y

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, (1-(1e-8)))

#洗牌函数，洗乱数据集
def _shuffle(X, Y):
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X, Y, percentage):
    all_size = X.shape[0]
    valid_size = int(floor(all_size * percentage))
    X, Y = _shuffle(X, Y)
    X_valid, Y_valid = X[ : valid_size], Y[ : valid_size]
    X_train, Y_train = X[valid_size:], Y[valid_size:]
    return X_train, Y_train, X_valid, Y_valid

def valid(X, Y, w):
    a = np.dot(w,X.T)
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y) == y_)
    acc = float(result.sum()) / result.shape[0]
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return y_ , acc

def train(X_train, Y_train):
    valid_set_percentage = 0.2
    w = np.zeros(len(X_train[0]))
    l_rate = 0.001
    batch_size = 32
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_train, Y_train, valid_set_percentage)
    train_dataz_size = len(X_train)
    step_num = int(floor(train_dataz_size / batch_size))
    epoch_num = 300
    list_cost = []
    list_cost_v = []
    accs_train = []
    accs_valid = []

    for epoch in range(1, epoch_num):
        total_loss = 0.0
        total_loss_v = 0.0
        #X_train, Y_train = _shuffle(X_train, Y_train)

        for idx in range(1, step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]
            z = np.dot(X, w)
            y = sigmoid(z)  #使用到了激活函数。
            grad = np.sum(-1 * X * (np.squeeze(Y) - y).reshape((batch_size, 1)), axis=0)
            w = w - l_rate * grad
            cross_entropy = -1 * (
                        np.dot(np.squeeze(Y.T), np.log(y)) + np.dot((1 - np.squeeze(Y.T)), np.log(1 - y))) / len(Y)
            total_loss += cross_entropy
            z_v = np.dot(X_valid, w)
            y_v = sigmoid(z_v)
            total_loss_v += -1 * (np.dot(np.squeeze(y_v.T), np.log(y_v)) + np.dot((1 - np.squeeze(y_v.T)),
                                                                                    np.log(1 - y_v))) / len(y_v)
        list_cost.append(total_loss)
        list_cost_v.append(total_loss_v)

        result = valid(X_train, Y_train, w)
        result_v = valid(X_valid, Y_valid, w)
        accs_train.append(result[1])
        accs_valid.append(result_v[1])

    drawLoss(list_cost,list_cost_v)
    drawAccs(accs_train,accs_valid)

    return w

def drawLoss(list_cost,list_cost_v):
    plt.figure()
    plt.plot(np.arange(len(list_cost)), list_cost)
    plt.plot(np.arange(len(list_cost_v)), list_cost_v)
    plt.legend(['train','dev'])
    plt.title("Train Process")
    plt.xlabel("epoch_num")
    plt.ylabel("Cost Function (Cross Entropy)")
    plt.savefig(os.path.join(os.path.dirname(output_dir), "TrainProcess"))
    plt.show()

def drawAccs(accs_train,accs_valid):
    plt.figure()
    plt.plot(np.arange(len(accs_train)), accs_train)
    plt.plot(np.arange(len(accs_valid)), accs_valid)
    plt.legend(['train','dev'])
    plt.title("Train Process")
    plt.xlabel("epoch_num")
    plt.ylabel("Accuracy of Function ")
    plt.savefig(os.path.join(os.path.dirname(output_dir), "TrainProcess_accuracy"))
    plt.show()

if __name__ == "__main__":
    trainData = pd.read_csv("data/train.csv")
    testData = pd.read_csv("data/test.csv")

    # here is one more attribute in trainData
    x_train = dataProcess_X(trainData).drop(['native_country_ Holand-Netherlands'], axis=1).values
    x_test = dataProcess_X(testData).values
    y_train = dataProcess_Y(trainData).values
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
    x_train = np.concatenate((np.ones((x_train.shape[0], 1)),x_train), axis=1)
    w = train(x_train, y_train)
    a = np.dot(w, x_test.T)
    y = sigmoid(a)
    y_ = np.around(y)
    df = pd.DataFrame({"id": np.arange(1, 16282), "label": y_})
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir + 'LR_output.csv'), sep='\t', index=False)



