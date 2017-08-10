# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:00:12 2017

@author: Administrator
"""

import csv
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


ep=0
base_number=[1,5,10,100]
kf=KFold(n_splits=10)
X=np.genfromtxt('data.csv',delimiter=',',dtype=np.float)
y=np.genfromtxt('targets.csv',delimiter=',',dtype=np.int)
X_scaled=preprocessing.scale(X)

#adboost train
clfs=[[],[],[],[]]
at=[[],[],[],[]] 
k=0
for i in base_number:    
    D=np.ones(len(X_scaled))/len(X_scaled)    
    for j in range(i):        
        clfs[k].append(LogisticRegression(C=10,solver='newton-cg',max_iter=10))
        clfs[k][j].fit(X_scaled,y,D)
        pred=clfs[k][j].predict(X_scaled)
        ep=1-clfs[k][j].score(X_scaled,y,D)
        if ep>0.5:
            break
        at[k].append(0.5*math.log( (1-ep)/ep) )      
        for x in range(len(X_scaled)):
            if pred[x]!=y[x]:
                D[x]*=np.exp(at[k][j])
            else:
                D[x]*=np.exp(-at[k][j])
        D=D/sum(D)
    k=k+1

#predict
for i in range(4):
    k=1
    t=base_number[i]      
    for train_index,test_index in kf.split(X_scaled,y):  
        csvfile = open('experiments/base%d_fold%d.csv'%(t,k), 'w',newline='')
        writer = csv.writer(csvfile)
        
        #mix predict
        for x in test_index:
            a=0       
            for q in range(len(at[i])):
                a=a+clfs[i][q].decision_function(X_scaled[x].reshape(1,-1))[0]*at[i][q]
            if a>0:
                writer.writerow( [ x+1,1 ])
            else:
                writer.writerow( [ x+1,0 ])
            
        csvfile.close()    
        k=k+1