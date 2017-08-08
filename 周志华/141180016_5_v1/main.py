# -*- coding: utf-8 -*-
"""
Created on Thu May 25 06:25:25 2017

@author: Administrator
"""
import pickle
import csv
import numpy as np

alpha=0.5

def cal(mean,variance,x):
    return np.log(1/np.sqrt(2*np.pi)/variance)-(x-mean)**2  /(2*variance**2)


def NaiveBayes(X,y):
    #calculate prior probility 
    n=np.zeros(5)
    for i in y:
        n[i]=n[i]+1
    p_prior=[]
    p_prior.append((n[0]+alpha)/(y.shape[0]+alpha*5))
    p_prior.append((n[1]+alpha)/(y.shape[0]+alpha*5))
    p_prior.append((n[2]+alpha)/(y.shape[0]+alpha*5))
    p_prior.append((n[3]+alpha)/(y.shape[0]+alpha*5))
    p_prior.append((n[4]+alpha)/(y.shape[0]+alpha*5))
    
    #calculate class conditional probability
    p_condition=np.zeros((5,2500))
    mean=np.zeros((5,2500))
    variance=np.zeros((5,2500))
    for i in range(X.shape[0]):
        x=np.array(X[i])[0]
        for j in range(2500):
            if x[j]==1:
                p_condition[y[i]][j]=p_condition[y[i]][j]+1      
                
    for j in range(2500,5000):
        m=[[],[],[],[],[]]
        for i in range(X.shape[0]):
            x=np.array(X[i])[0] 
            m[y[i]].append(x[j])
        for i in range(5):
            mean[i][j-2500]=np.mean(m[i])     
            variance[i][j-2500]=np.var(m[i])
            
    for i in range(5): 
        for j in range(2500):
            p_condition[i][j]=(p_condition[i][j]+alpha)/(n[i]+alpha*2)       
    
    return p_prior,p_condition,mean,variance
    
    
    
def predict(Xt,p_prior,p_condition,mean,variance):    
    csvfile = open('test_predictions.csv','w',newline='')
    writer = csv.writer(csvfile)
    
    for x in Xt:
        x=np.array(x)[0]
        p=[]
        for label in range(5):
            a=0
            for j in range(2500):
                if x[j]==1:
                    a=a+np.log(p_condition[label][j])
                else:
                    a=a+np.log(1-p_condition[label][j])
#            for j in range(2500,5000):
#                if variance[label][j-2500] != 0:
#                    b=cal(mean[label][j-2500],variance[label][j-2500],x[j])
#                    a=a+b
            p.append(np.log(p_prior[label])+a)
        writer.writerow([p.index(max(p))])
        
    csvfile.close()       
                

X=pickle.load(open('train_data.pkl','rb')).todense() # unsupported in Python 2
y=pickle.load(open('train_targets.pkl','rb'))
Xt=pickle.load(open('test_data.pkl','rb')).todense()
p_prior,p_condition,mean,variance=NaiveBayes(X,y)
predict(Xt,p_prior,p_condition,mean,variance)