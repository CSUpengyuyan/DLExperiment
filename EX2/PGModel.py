import pandas as pd
import numpy as np
from random import shuffle
from numpy.linalg import inv
import os


def dataProcess_X(rawData):
    #sex 只有两个属性 先drop之后处理
    if "income" in rawData.columns:
        Data = rawData.drop(["sex", 'income'], axis=1) #删掉性别，收入两行
    else:
        Data = rawData.drop(["sex"], axis=1)
    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"] #读取非数字的column
    listNonObjedtColumn = [x for x in list(Data) if x not in listObjectColumn] #数字的column
    #提出那些列是对象的拿出来。
    ObjectData = Data[listObjectColumn]
    NonObjectData = Data[listNonObjedtColumn]
    #insert set into nonobject data with male = 0 and female = 1
    NonObjectData.insert(0 ,"sex", (rawData["sex"] == " Female").astype(np.int))# 首先 0 列，列名：sex，第三个参数：数值， 0代表男性，1代表女性。
    #set every element in object rows as an attribute
    ObjectData = pd.get_dummies(ObjectData)# 独热编码。

    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data_x = Data.astype("int64")
    # Data_y = (rawData["income"] == " <=50K").astype(np.int)

    #normalize
    Data_x = (Data_x - Data_x.mean()) / Data_x.std() #每列求平均值，然后计算与平均值之间的距离 / 标准偏差。
    #返回的是经过处理过的dataframe。独热编码
    return Data_x
def dataProcess_Y(rawData):
    df_y = rawData['income']
    Data_y = pd.DataFrame((df_y==' >50K').astype("int64"), columns=["income"]) #不知道多少行，一列的数据。（n,1)
    return Data_y

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, (1-(1e-8))) #防止返回的数过小，过大

# def _shuffle(X, Y):                                 #X and Y are np.array
#     randomize = np.arange(X.shape[0])
#     np.random.shuffle(randomize)
#     return (X[randomize], Y[randomize])
#
# 把x，y分开成训练集和验证集。
# def split_valid_set(X, Y, percentage):
#     all_size = X.shape[0]
#     valid_size = int(floor(all_size * percentage))
#
#     X, Y = _shuffle(X, Y)
#     X_valid, Y_valid = X[ : valid_size], Y[ : valid_size]
#     X_train, Y_train = X[valid_size:], Y[valid_size:]
#
#     return X_train, Y_train, X_valid, Y_valid

# # 验证的函数
# def valid(X, Y, mu1, mu2, shared_sigma, N1, N2):
#     sigma_inv = inv(shared_sigma)
#     w = np.dot((mu1-mu2), sigma_inv)
#     X_t = X.T
#     b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(float(N1)/N2)
#     a = np.dot(w,X_t) + b
#     y = sigmoid(a)
#     y_ = np.around(y)
#     result = (np.squeeze(Y) == y_)
#     print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
#     return

def train(X_train, Y_train):
    # vaild_set_percetange = 0.1
    # X_train, Y_train, X_valid, Y_valid = split_valid_set(X, Y, vaild_set_percetange)

    #Gussian distribution parameters
    train_data_size = X_train.shape[0]

    num1 = 0
    num2 = 0

    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))
    for i in range(train_data_size):
        if Y_train[i] == 1:     # >50k
            mu1 += X_train[i]
            num1 += 1
        else:
            mu2 += X_train[i]
            num2 += 1
# mu1 present >50k  平均值mu1、mu2
    mu1 /= num1
    mu2 /= num2
    sigma1 = np.zeros((106, 106))
    sigma2 = np.zeros((106, 106))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [X_train[i] - mu2])

    sigma1 /= num1 #平均方差
    sigma2 /= num2
    #共享方差
    shared_sigma = (float(num1) / train_data_size) * sigma1 + (float(num2) / train_data_size) * sigma2
    N1 = num1
    N2 = num2
    sigma_inv = inv(shared_sigma)
    # 两个函数之间的差 * 公共方差
    w = np.dot((mu1 - mu2), sigma_inv)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(
        float(N1) / N2)
# 训练出最终的w,b
    return w,b

if __name__ == "__main__":
    trainData = pd.read_csv("data/train.csv")
    testData = pd.read_csv("data/test.csv")
#  ans = pd.read_csv("data/correct_answer.csv")sender

# here is one more attribute in trainData #去除杂数据
    x_train = dataProcess_X(trainData).drop(['native_country_ Holand-Netherlands'], axis=1).values
    print(x_train.shape)
    x_test = dataProcess_X(testData).values
    print(x_test.shape)
    y_train = dataProcess_Y(trainData).values
 #   y_ans = ans['label'].values
 #   vaild_set_percetange = 0.1
 #   X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y_train, vaild_set_percetange)
    w,b = train(x_train, y_train)
#valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)
# mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)
# 矩阵的求逆
    X_t = x_test.T
    a = np.dot(w, X_t) + b
    y = sigmoid(a)
    y_ = np.around(y).astype(np.int)
   # df = pd.DataFrame({"id" : np.arange(1,16282), "label": y_})
   #  result = (np.squeeze(y_ans) == y_)
   #  print('Test acc = %f' % (float(result.sum()) / result.shape[0]))
    # 输出结果到output文件里面
    output_dir = "output/"
    df = pd.DataFrame({"id": np.arange(1, 16282), "label": y_})
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir+'PGModelPredict.csv'), sep='\t', index=False)
