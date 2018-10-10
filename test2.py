# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:10:33 2018

@author: 25091
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:38:34 2018

@author: 25091
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:24:03 2018

@author: 25091
"""

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
#from tensorflow.contrib import keras 

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
# INSERT YOUR CODE HERE
#np.savez('D:\\mnist_dataset.npz',x_train,y_train,x_test,y_test)

list1=[]
for j in [0,1,8,9]:
    location=0
    for i in y_train:
        if i==j:
            list1.append(location)
        location=location+1


x_train1=np.delete(x_train, list1, 0)
y_train1=np.delete(y_train,list1,0)
x_test=np.delete(x_test,list1,0)
y_test=np.delete(y_test,list1,0)

img_rows, img_cols = x_train.shape[1:3]
X_train =  x_train1.reshape(x_train1.shape[0], img_rows, img_cols, 1) #将图片向量化
X_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
x_train1=X_train/255 # 归一化
x_test=X_test/255


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = [] #一会儿一对对的样本要放在这里
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)])-1
    for d in range(10):
        #对第d类抽取正负样本
        for i in range(n):
            # 遍历d类的样本，取临近的两个样本为正样本对
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            # randrange会产生1~9之间的随机数，含1和9
            inc = np.random.randint(1, 10)
            # (d+inc)%10一定不是d，用来保证负样本对的图片绝不会来自同一个类
            dn = (d + inc) % 10
            # 在d类和dn类中分别取i样本构成负样本对
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            # 添加正负样本标签
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_pairs1(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = [] #一会儿一对对的样本要放在这里
    labels = []
    n = min([len(digit_indices[d]) for d in range(0,5)])-1
    print(n)
    for d in range(0,5):
        #对第d类抽取正负样本
        for i in range(n):
            # 遍历d类的样本，取临近的两个样本为正样本对
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            # randrange会产生1~9之间的随机数，含1和9
            inc = np.random.randint(2, 7)
            # (d+inc)%10一定不是d，用来保证负样本对的图片绝不会来自同一个类
            dn = (d + inc) % 6
            # 在d类和dn类中分别取i样本构成负样本对
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            # 添加正负样本标签
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = keras.Sequential()
    seq.add(keras.layers.Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(keras.layers.Dropout(0.1))
    seq.add(keras.layers.Dense(128, activation='relu'))
    seq.add(keras.layers.Dropout(0.1))
    seq.add(keras.layers.Dense(128, activation='relu'))
    return seq

def create_CNN_network(input_dim):
    seq=keras.Sequential()
    
    seq.add(keras.layers.Conv1D(input_shape=(input_dim,),filters=6,kernel_size=(5,5),activation='relu'))
    seq.add(keras.layers.Conv2D(filters=12,kernel_size=(3,3),activation='relu'))
    seq.add(keras.layers.MaxPooling2D(poolling_size=(3,3)))
    seq.add(keras.layers.Flatten())

    seq.add(keras.layers.Dropout(0.1))
    seq.add(keras.layers.Dense(128, activation='relu'))
    seq.add(keras.layers.Dropout(0.1))
    seq.add(keras.layers.Dense(128, activation='relu'))
    return seq


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)




digit_input = keras.layers.Input(shape=input_shape)
x = keras.layers.Conv2D(32,(5, 5))(digit_input)
x = keras.layers.Conv2D(64,(3, 3))(x)
#x = keras.layers.Conv2D(12,(3, 3))(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Dropout(0.25)(x)
out =keras.layers.Flatten()(x)

vision_model = keras.models.Model(digit_input, out)
#base_network = create_CNN_network(784)

input_a = keras.layers.Input(shape=input_shape, name = 'input_a')
input_b = keras.layers.Input(shape=input_shape, name = 'input_b')

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = vision_model(input_a)
processed_b = vision_model(input_b)

#distance = keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

concatenated = keras.layers.concatenate([processed_a, processed_b], axis=-1)
out1 = keras.layers.Dense(128, activation='relu')(concatenated)
out1 = keras.layers.Dropout(0.5)(out1)
out1 = keras.layers.Dense(64, activation='relu')(out1)
out1 = keras.layers.Dropout(0.25)(out1)
out = keras.layers.Dense(1, activation='sigmoid')(out1)
model = keras.models.Model([input_a, input_b],out)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


digit_indices1 = [np.where(y_train1 == i)[0] for i in [2,3,4,5,6,7]]
tr_pairs, tr_y = create_pairs1(x_train1, digit_indices1)
#tr_y=np.delete(tr_y,[0,1,8,9],0)
digit_indices2 = [np.where(y_test == i)[0] for i in [2,3,4,5,6,7]]
te_pairs, te_y = create_pairs1(x_test, digit_indices2)

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

rms = keras.optimizers.RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms,metrics=['accuracy'])
AAA=model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              epochs = 10,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
              batch_size=128)



acc = AAA.history['acc']
val_acc = AAA.history['val_acc']
loss = AAA.history['loss']
val_loss = AAA.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()








# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred.round(), te_y.round())


print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))