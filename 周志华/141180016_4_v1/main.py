# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:09:23 2017

@author: Administrator
"""
import numpy as np

#from sklearn.svm import SVC

def predict(Xt,model):
    result=[]

    i=0
    for item in Xt:
        sums=0
        j=0
        for supports in model.support_vectors_:
            mult=sum((item-supports)*(item-supports))
            k=np.exp(-mult*model.gamma)
            sums+=model.dual_coef_[0][j]*k
            j+=1
            
        sums+=model.intercept_
        if sums>0:
            result.append(1)
        else:
            result.append(0)
        i+=1
    return result