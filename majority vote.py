# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 00:37:54 2019

@author:  tongwang
"""

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer, f1_score, recall_score

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

data=pd.read_csv('dataset for ensemble.csv')
predict=data['mean_6']
ture=data['true_labels']
def roc_auc_func(y_true, y_score):
    return roc_auc_score(y_true, y_score, average='weighted')
print('Test auc score %s' %roc_auc_func(ture, predict))
def print_res(fpr, tpr, thresholds, class_1_min_recall=.60):
    sens  = tpr[tpr>=class_1_min_recall]
    specs = 1 - fpr[tpr>=class_1_min_recall]
    thres = thresholds[tpr>=class_1_min_recall]

    print("Threshold: ")
    print(thres[0])

    print("Recall class 0:")
    print(specs[0])


    print("Recall class 1:")
    print(sens[0])
    
fpr, tpr, thresholds = roc_curve(ture, predict, pos_label=1) 
for class_1_min_recall in [0.5,	0.51,	0.52,	0.53,	0.54,	0.55,	0.56,	0.57,	0.58	,0.59	,0.6,	0.61,	0.62,	0.63,	0.64,	0.65,	0.66,	0.67	,0.68,	0.69,	0.7,	0.71,	0.72,	0.73	,0.74	,0.75,	0.76,	0.77,	0.78,	0.79,	0.8,	0.81,	0.82,	0.83,	0.84,	0.85,	0.86,	0.87,	0.88,	0.89,	0.9,	0.91,	0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99
]:
    print("=============================")
    print("Sens >= %s :" % class_1_min_recall)
    print_res(fpr, tpr, thresholds, class_1_min_recall)
    print()
#如果要得到confusion matrix, 需要的是TRUE label和predicted label 而不是概率，但是如果用的不是概率，得到的auc就会小一点。
predict=np.array(predict)
for i in range(len(predict)):
    if predict[i]>=0.5:
        predict[i]=1
    else:
        predict[i]=0        
print(classification_report(ture,predict))


