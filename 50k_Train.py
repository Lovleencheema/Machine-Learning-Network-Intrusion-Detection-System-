'''
Dos attack detection using svm clasifier.
Dataset contains 2300000 records.
Approah Train/Test Split.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
import random
from mlxtend.plotting import plot_decision_regions
import pickle
from Data import *

print("Total records in Dataset \n",finalDataset.count())
print("Total records in Training Dataset \n",X_train.count())
print("Total records in Testing Dataset: \n",X_test.count())

# number of folds
num_folds = 10
# creating standard scaler.
scaler = StandardScaler().fit(X_train)
#Rescaling the scaler.
rescaledX = scaler.transform(X_train)
# value of C.
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
# types of kernel.
kernel_values = ['linear', 'rbf', 'sigmoid']
# dictnory of c values and kernal types.
param_grid = dict(C=c_values, kernel=kernel_values)
# creating model.
model = SVC()
# creating folds with 21 random state.
kfold = KFold(n_splits=num_folds, random_state=21)
# creating grid.
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
# training the model.
print("Traning started ... ")
grid_result = grid.fit(rescaledX, y_train)
print("Traning Done! ")
# calculating means of test score.
means = grid_result.cv_results_['mean_test_score']
# calculating std. of test score.
stds = grid_result.cv_results_['std_test_score']
# list of params.
params = grid_result.cv_results_['params']
#printing the result of each combination.   
for mean, stdev, param in zip(means, stds, params):
    print("-----------------------------------------------------------")
    print("|%f (%f) with: %r" % (mean, stdev, param),"|")
# print the best score and combination.
print("\n\n=================================================================")
print(" The best score is: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("=================================================================")

filename = 'ddos_attack_detector_new.sav'
pickle.dump(grid_result, open(filename, 'wb'))
