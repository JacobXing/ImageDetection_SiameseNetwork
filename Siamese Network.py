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

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

num_classes = 10
num_train = 50000
num_test = 10000
validation_split = 0.1
val_idx = -10000
epochs = 20
normalize = True
test_only_digits = [0,1,8,9]
training_digits = [cls for cls in range(num_classes) if cls not in test_only_digits]
margin = 1
threshould = margin/2

def create_pairs(x, digits, num_pairs, digits2 = None):
    
    pairs = []
    labels = []
            
    if digits2 is None:
            
        while True: # Change to while True? 
            
            for d in digits:
                
                P1, P2 = np.random.choice(digits_idx[d],2)
                pairs += [[x[P1], x[P2]]]

                assert y[P1] in digits and y[P2] in digits
                assert y[P1] == y[P2], 'Positive pairs should have the same labels'

                N1 = np.random.choice(digits_idx[d])
                N2 = np.random.choice(digits_idx[np.random.choice([ di for di in digits if di != d])])
                pairs += [[x[N1], x[N2]]]

                assert y[N1] in digits and y[N2] in digits
                assert y[N1] != y[N2], 'Negative pairs should have different labels'

                labels += [1, 0]
                
                if len(pairs) >= num_pairs:
                    if normalize:
                        return np.array(pairs).astype('float32')/255, np.array(labels)
                    else:
                        return np.array(pairs).astype('float32'), np.array(labels)
    else:
        
        while True: 

            d1 = np.random.choice(digits)
            d2 = np.random.choice(digits2)

            P1 = np.random.choice(digits_idx[d1])
            P2 = np.random.choice(digits_idx[d2])

            pairs += [[x[P1], x[P2]]]

            labels += [d1 == d2]

            if len(pairs) >= num_pairs:
                
                if normalize:
                    return np.array(pairs).astype('float32')/255, np.array(labels)
                else:
                    return np.array(pairs).astype('float32'), np.byte(np.array(labels))

def create_siameseNetwork(input_shape):
    digit_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(32,(5, 5),activation='relu')(digit_input)
    x = keras.layers.Conv2D(64,(3, 3),activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)
    out =keras.layers.Flatten()(x)

    vision_model = keras.models.Model(digit_input, out)

    input_a = keras.layers.Input(shape=input_shape, name = 'input_a')
    input_b = keras.layers.Input(shape=input_shape, name = 'input_b')

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
    processed_a = vision_model(input_a)
    processed_b = vision_model(input_b)

    concatenated = keras.layers.concatenate([processed_a, processed_b], axis=-1)
    out1 = keras.layers.Dense(128, activation='relu')(concatenated)
    out1 = keras.layers.Dropout(0.5)(out1)
    out1 = keras.layers.Dense(64, activation='relu')(out1)
    out1 = keras.layers.Dropout(0.25)(out1)
    out = keras.layers.Dense(1, activation='sigmoid')(out1)
    model = keras.models.Model([input_a, input_b],out)
    return model

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    '''
    tmp= y_true *tf.square(y_pred)
    tmp2 = (1-y_true) *tf.square(tf.maximum((1 - y_pred),0))
    return (tf.reduce_sum(tmp/2) +tf.reduce_sum(tmp2/2))


def compute_accuracy(predictions, labels, indices=True):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < threshould].mean()

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < threshould, y_true.dtype)))
def evaluate_accuracy(model, x, digits1, num_pairs, digits2 = None):
    
    X_eval, y_eval = create_pairs(x, digits1, num_pairs, digits2)
    y_pred = model.predict(x = [X_eval[:, 0], X_eval[:, 1]])
    eval_acc = compute_accuracy(y_eval, y_pred)
    if digits2 is None:
        return {'[%s]x[%s]'% (digits1, digits1): eval_acc}
    else:
        return {'[%s]x[%s]'% (digits1, digits2): eval_acc}

def evaluation_Statistic_Result(model, x, digits1, num_pairs, digits2 = None):
# compute final accuracy on training and test sets
    X_eval, y_eval = create_pairs(x, digits1, num_pairs, digits2)
    y_pred = model.predict(x = [X_eval[:, 0], X_eval[:, 1]])

    for i in range(len(y_pred)):
        if y_pred[i] < threshould:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    labels = ['different image', 'same image']
    if digits2 is None:
        print('\n[%s]x[%s]:'% (digits1, digits1))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_pred, y_eval)
        import pandas as pd
        cm = pd.DataFrame(cm)
        cm.columns = labels
        cm.index = labels
        print('\nConfusion_matrix:')
        print(cm)
        from sklearn.metrics import classification_report
        print('\nClassification report:')
        print(classification_report(y_eval, y_pred, target_names=labels)) 
    else:
        print('\n[%s]x[%s]:'% (digits1, digits2))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_pred, y_eval)
        import pandas as pd
        cm = pd.DataFrame(cm)
        cm.columns = labels
        cm.index = labels
        print('\nConfusion_matrix:')
        print(cm)
        from sklearn.metrics import classification_report
        print('\nClassification report:')
        print(classification_report(y_eval, y_pred, target_names=labels))

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

X = np.expand_dims(np.concatenate((x_train,x_test), axis= 0),1)
y = np.concatenate((y_train,y_test), axis= 0)
img_rows, img_cols = X.shape[2:]
X =  X.reshape(X.shape[0], img_rows, img_cols, 1) 

digits_idx = [np.where(y == i)[0] for i in range(num_classes)]

X_train, y_train = create_pairs(X, training_digits, num_train)

input_shape = X_train.shape[2:]
if val_idx is None:
    val_idx = -(validation_split) * len(X_train)


model = create_siameseNetwork(input_shape)
rms = keras.optimizers.RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
AAA = model.fit([X_train[:val_idx, 0], X_train[:val_idx, 1]], y_train[:val_idx],
                batch_size=128,
                validation_data=([X_train[val_idx:, 0], X_train[val_idx:, 1]], y_train[val_idx:]),
                epochs=epochs)



acc = AAA.history['accuracy']
val_acc = AAA.history['val_accuracy']
loss = AAA.history['loss']
val_loss = AAA.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training loss')
plt.plot(epochs, val_loss, '-', color='blue', label='validation loss')
plt.title('Training and validation error VS time')
plt.legend()
plt.show()

eval_table = {}
y_pred = model.predict([X_train[:val_idx, 0], X_train[:val_idx, 1]])
train_acc = compute_accuracy(y_train[:val_idx], y_pred)
eval_table.update({'Train': train_acc})
#eval_table.update(evaluate_accuracy(model, X, range(num_classes), num_test))
eval_table.update(evaluate_accuracy(model, X, training_digits, num_test))
eval_table.update(evaluate_accuracy(model, X, training_digits, num_test, test_only_digits))
eval_table.update(evaluate_accuracy(model, X, test_only_digits, num_test))
import pandas as pd
print(pd.DataFrame([eval_table]).T)


# compute final accuracy on training and test sets
y_pred = model.predict([X_train[:val_idx, 0], X_train[:val_idx, 1]])
tr_acc = compute_accuracy(y_train[:val_idx], y_pred)
y_pred = model.predict([X_train[val_idx:, 0], X_train[val_idx:, 1]])
te_acc = compute_accuracy(y_train[val_idx:], y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
evaluation_Statistic_Result(model, X, training_digits, num_test)
evaluation_Statistic_Result(model, X, training_digits, num_test, test_only_digits)
evaluation_Statistic_Result(model, X, test_only_digits, num_test)