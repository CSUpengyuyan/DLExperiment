# import numpy as np
# #自行创建的简单数据
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#
# # 构建数据中所有标记的索引，用一个字典来存储
# token_index = {}
# for sample in samples:
#     # 利用split方法对样本进行分词.
#     for word in sample.split():
#         if word not in token_index:
#             #为每个唯一单词指定一个唯一索引
#             token_index[word] = len(token_index) + 1
#             #没有为索引编号0指定单词
#
# #对样本进行分词
# # 只考虑每个样本前max_length个单词。
# max_length = 10
#
# #结果返回给results:
# results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = token_index.get(word)
#         #指定唯一的元素为1
#         results[i, j, index] = 1.
# #查看索引字典
# print(token_index)
# print(results[1,1])#样本列表的第二个元素的第二个单词编码情况



# import numpy as np
# #自行创建的简单数据
# samples = ['The cat sat on the mat.', 'The dog ate my homework.','a panda is sleeping.']
#
# # 构建数据中所有标记的索引，用一个字典来存储
# token_index = {}
# for sample in samples:
#     # 利用split方法对样本进行分词.
#     for word in sample.split():
#         if word not in token_index:
#             #为每个唯一单词指定一个唯一索引
#             token_index[word] = len(token_index) + 1
#             #没有为索引编号0指定单词
#
# #对样本进行分词
# # 只考虑每个样本前max_length个单词。
# max_length = 10
#
# #结果返回给results:
# results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = token_index.get(word)
#         #唯一的元素为1
#         results[i, j, index] = 1.
# #查看索引字典
# print(token_index)
# print(results[1,1])#样本列表的第二个元素的第二个单词编码情况


import numpy as np
# import string
#
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# characters = string.printable  #所以可以打印的ASCII字符
# #创建索引字典
# token_index = dict(zip(characters, range(1, len(characters) + 1)))
# #为所有可能打印的字符创建一个字典
# max_length = 50
# results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
# for i, sample in enumerate(samples):
#     for j, character in enumerate(sample[:max_length]):
#         index = token_index.get(character)
#         results[i, j, index] = 1.
# print(token_index)#查看索引字典
# print(results[1,1])#样本列表的第二个元素的第二个字符编码情况



# import string
#
# samples = ['The cat sat on the mat.', 'The dog ate my homework.','a panda is sleeping.']
# characters = string.printable  #所以可以打印的ASCII字符
# #创建索引字典
# token_index = dict(zip(characters, range(1, len(characters) + 1)))
# #为所有可能打印的字符创建一个字典
# max_length = 50
# results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
# for i, sample in enumerate(samples):
#     for j, character in enumerate(sample[:max_length]):
#         index = token_index.get(character)
#         results[i, j, index] = 1.
# print(token_index)#查看索引字典
# print(results[2,2])#样本列表的第3个元素的第3个字符编码情况



# from keras.preprocessing.text import Tokenizer
#
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#
# #创建一个分词器
# # 只考虑前1000个最常见的单词
# tokenizer = Tokenizer(num_words=1000)
# # 构建单词索引
# tokenizer.fit_on_texts(samples)
#
# # 将字符串转换为整数索引的组成的列表
# sequences = tokenizer.texts_to_sequences(samples)
#
# # 可以直接得到one-hot编码二进制表示
# # 分词器也支持除one-hot编码外的其他向量化模式
# one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
#
# # 找回单词索引
# word_index = tokenizer.word_index
# print(word_index)




from keras.layers import Embedding

# Embedding层至少需要2个参数
# 标记的个数（这里是1000，即最大单词索引+1）和嵌入维度（这里是64）
embedding_layer = Embedding(1000, 64)


from keras.datasets import imdb
from keras import preprocessing
# 作为特征的单词，即选取出现频率最高的单词数量
max_features = 10000
# 在这么多单词后截断文本
# (这些单词都属于前max_features个最常见单词)
maxlen = 20
#将数据集加载为列表
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# 将整数列表转换成形状为（samples,maxlen）的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# 指定Embedding层的最大输入长度，以便后面将嵌入输入展平。
model.add(Embedding(10000, 8, input_length=maxlen))
# 将三维的嵌入张量展平为(samples, maxlen * 8)的二维张量
model.add(Flatten())

# 添加分类器
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)