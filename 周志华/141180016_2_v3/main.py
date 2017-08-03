# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:13:37 2017

@author: Administrator
"""
import csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
def P1(input):
    return 1/(1+np.exp(-input))

    
def FirstOrder(X,y,weights):
    m=X.shape[0]
    n=X.shape[1]
    sum_gd = np.matrix(np.zeros([1,n])).T
    for i in range(m):
        x_i = np.matrix(X[i]).T
        p1 = P1((weights * x_i).item(0, 0))
        sum_gd += (p1 - y[i]) * x_i
    return sum_gd
    
    
def SecondOrder(X,y,weights):    
    m=X.shape[0]
    n=X.shape[1]
    sum_hs = np.matrix(np.zeros([n, n]))
    for i in range(m):
        x_i = np.matrix(X[i]).T
        p1 = P1((weights * x_i).item(0, 0))
        sum_hs += x_i * x_i.T*p1*(1-p1)
    return sum_hs


def NewtonMethod(X,y,weights):
    a=SecondOrder(X,y,weights)
    b=FirstOrder(X,y,weights)
    weights=weights-np.dot(np.linalg.inv(a),b).T
    return weights

k=0
X=np.genfromtxt('data.csv',delimiter=',',dtype=np.float)
y=np.genfromtxt('targets.csv',delimiter=',',dtype=np.int)
#generalization    
X_scaled=preprocessing.scale(X)

kf=KFold(n_splits=10)
result=list()
for train_index,test_index in kf.split(X,y):  
    k=k+1
    X_train,X_test=X_scaled[train_index],X_scaled[test_index]
    y_train,y_test=y[train_index],y[test_index]
    n=X_train.shape[1]
    m=X_test.shape[0]    
    csvfile = open('fold%d.csv'%k, 'w',newline='')
    writer = csv.writer(csvfile)
    #data=list()
    
    weights=np.matrix(np.zeros(n))    
    for i in range(3):
        weights=NewtonMethod(X_train,y_train,weights)
    for i in range(m):
        if P1(weights*np.matrix(X_test[i]).T)>=0.5:
            writer.writerow([test_index[0]+i+1,'1'])
            #data.append((str((k-1)*56+i+1),'1'))
        else:
            writer.writerow([test_index[0]+i+1,'0'])
            #data.append((str((k-1)*56+i+1),'0'))
    #writer.writerow(data)
    csvfile.close()       
            
        
    

