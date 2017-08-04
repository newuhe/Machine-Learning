# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:07:30 2017

@author: Administrator
"""

import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

X_train=np.matrix(np.genfromtxt('train_data.csv',delimiter=',',dtype=np.float))
y_train=np.matrix(np.genfromtxt('train_targets.csv',delimiter=',',dtype=np.int)).T
train_labels=to_categorical(y_train,num_classes=10)
X_test=np.matrix(np.genfromtxt('test_data.csv',delimiter=',',dtype=np.float))
y_test=np.matrix(np.genfromtxt('test_targets.csv',delimiter=',',dtype=np.int))
test_labels=to_categorical(y_test,num_classes=10)
accuracy=0

model = Sequential()
model.add(Dense(512, input_dim=400,init='uniform'))
model.add(Activation('relu'))
model.add(Dense(512, input_dim=512))
model.add(Activation('relu'))
model.add(Dense(10)) #输出层
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd')

#while accuracy<0.93:
model.fit(X_train, train_labels, nb_epoch = 20, batch_size = 1)
    
a=model.predict_classes(X_test)    
csvfile = open('test_predictions_library.csv', 'w',newline='')
writer = csv.writer(csvfile)
for i in range(a.shape[0]):
    writer.writerow(str(a[i]))
csvfile.close() 
    