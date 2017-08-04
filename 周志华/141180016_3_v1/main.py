# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:04:18 2017

@author: Administrator
"""
import csv
import numpy as np
#import matplotlib.pyplot as plt
X_train=np.matrix(np.genfromtxt('train_data.csv',delimiter=',',dtype=np.float))
y_train=np.matrix(np.genfromtxt('train_targets.csv',delimiter=',',dtype=np.int)).T
X_test=np.matrix(np.genfromtxt('test_data.csv',delimiter=',',dtype=np.float))
#plt.imshow(x.reshape((20,20),order='F'),cmap='gray')
#plt.show()

def Sigmoid(x):
    return 1.0/(1+np.exp(-x))

    
lamda=0.1   
W=np.matrix(0.1*np.random.rand(10,100)-0.05)
theta2=np.matrix(np.zeros([1,10])).T
V=np.matrix(0.1*np.random.rand(100,400)-0.05)
theta1=np.matrix(np.zeros([1,100])).T  
accuracy=0

for i in range(150):
    csvfile = open('test_predictions.csv', 'w',newline='')
    writer = csv.writer(csvfile)
    for i in range(X_train.shape[0]):
        alpha=V*np.matrix(X_train[i]).T
        b=Sigmoid(alpha-theta1)
        beta=W*b
        y=Sigmoid(beta-theta2)
        
        y_t=np.matrix(np.zeros([10,1]))
        y_t[y_train[i]]=1
        g=np.matrix(np.multiply(np.multiply(y,1-y),y_t-y))   
        dW=lamda*g*b.T
        dtheta2=-lamda*g
        e=np.matrix(np.multiply(np.multiply(b,1-b),W.T*g))
        dV=lamda*e*np.matrix(X_train[i])
        dtheta1=-lamda*e
        
        W=W+dW
        V=V+dV
        theta1=theta1+dtheta1
        theta2=theta2+dtheta2    

    for i in range(X_test.shape[0]):
        alpha=V*np.matrix(X_test[i]).T
        b=Sigmoid(alpha-theta1)
        beta=W*b
        y=Sigmoid(beta-theta2)
        position=np.argmax(y)
        writer.writerow([position])
  
      
