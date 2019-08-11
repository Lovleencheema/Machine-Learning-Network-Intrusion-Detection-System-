'''
Dos attack detection using svm clasifier.
Dataset contains 2300000 records.
Approah Train/Test Split.
'''
import time
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import random
from mlxtend.plotting import plot_decision_regions
import pickle
from Data import *
import warnings
warnings.simplefilter("ignore")

# function to select n random value.
def some(x, n):
    return x.ix[random.sample(x.index, n)]
# To specify the the type of window.
def specifier(n):
    if n == 0:
        return '0'
    else:
        return '1'

count = 0# count to identify terminate the while loop.
justkidding = 0
total_rows = X_test['Timestamp'].count()# lenth of the test data frame.
count1 = 0
while count1 < int(total_rows):
    PERIOD_OF_TIME = 0.25 # 1 second.
    start = time.time()# start timer.
    rcounter = 0
    print('|',end="")
    while True:
        rcounter = rcounter + 1# row counter
        count = count + 1
        if time.time() > start + PERIOD_OF_TIME:break # break after 12 sec.
        
        print('=',end="")# process.
    if justkidding == 0:
        intail = 0# starting index of window.
        last = rcounter # last index of window.
    else:
        temp = last
        intail = temp# starting index of window.
        last = temp+rcounter# last index of window.
        

    justkidding = justkidding +1
    if  last < int(total_rows): # while loop runs 12 sec.

        
        # load the model from disk
        svclassifier = pickle.load(open('ddos_attack_detector_new.sav', 'rb'))
        #result = loaded_model.score(X_test, Y_test)

        #testforNow = X_test.sample(frac=0.05)
        testforNow = X_test[intail:last]
        testyforNow =  y_test[intail:last]
        # predictions.
        y_pred = svclassifier.predict(testforNow)
        numberOfZeros = 0;
        NubnerOfNon = 0;
        # counting number of attack and normal msgs.
        for i in y_pred:
            if i == 0:
                numberOfZeros = numberOfZeros+1;
            else:
                NubnerOfNon = NubnerOfNon+1;
        # printing feature vector table.
        print('|')
        print('\nFeatures Vector:')
        print('-------------------------------------------------------------------------------------------------------------------------------')
        print('| No. of messages with normal ID | No. of messages with attack ID | Class(0 specifies- no attack 1 specifies- attack message) |')
        print('-------------------------------------------------------------------------------------------------------------------------------')
        print('|', NubnerOfNon," "*(29-len(str(NubnerOfNon))),'|',numberOfZeros,' '*(29-len(str(numberOfZeros))),'|',specifier(numberOfZeros),' '*(56-len(str(specifier(numberOfZeros)))),'|')
        print('-------------------------------------------------------------------------------------------------------------------------------')
        # Pllting a pie chart.
        labels = ['Normal Messages','Attack Messages']
        stat = [NubnerOfNon,numberOfZeros]
        colors = ['lightcoral','lightskyblue']
        plt.pie(stat,labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title('Features Vectors')
        plt.show()

        # ploting a bar chart.
        y = [NubnerOfNon,numberOfZeros]
        N = len(y)
        width = 1/4
        barlist = plt.bar(labels, y, width)
        barlist[0].set_color('g')
        barlist[1].set_color('r')
        plt.show()


        from sklearn.metrics import classification_report, confusion_matrix
        # confusion matrix.
        print('Confusion matrix: ',confusion_matrix(testyforNow,y_pred))
        # Confusion report.
        print('Confusion report: ',classification_report(testyforNow,y_pred))
        # accuracy.
        print('Accurany: ',svclassifier.score(testforNow, testyforNow))
    else:# if data ended before 12 sec.s
                
        # load the model from disk
        svclassifier = pickle.load(open('ddos_attack_detector_new.sav', 'rb'))
        #result = loaded_model.score(X_test, Y_test)

        #testforNow = X_test.sample(frac=0.05)
        testforNow = X_test[intail:int(total_rows)]
        testyforNow =  y_test[intail:int(total_rows)]
        # predictions.
        y_pred = svclassifier.predict(testforNow)
        numberOfZeros = 0;
        NubnerOfNon = 0;
        # counting number of attack and normal msgs.
        for i in y_pred:
            if i == 0:
                numberOfZeros = numberOfZeros+1;
            else:
                NubnerOfNon = NubnerOfNon+1;
        # printing feature vector table.
        print('|',end="\n")
        print('Feature Vector:')
        print('-------------------------------------------------------------------------------------------------------------------------------')
        print('| No. of messages with normal ID | No. of messages with attack ID | Class(0 specifies- no attack 1 specifies- attack message) |')
        print('-------------------------------------------------------------------------------------------------------------------------------')

        print('|', NubnerOfNon," "*(29-len(str(NubnerOfNon))),'|',numberOfZeros,' '*(29-len(str(numberOfZeros))),'|',specifier(numberOfZeros),' '*(56-len(str(specifier(numberOfZeros)))),'|')
        print('-------------------------------------------------------------------------------------------------------------------------------')
        # Ploting a pie chart.
        labels = ['Normal Messages','Attack Messages']
        stat = [NubnerOfNon,numberOfZeros]
        colors = ['lightcoral','lightskyblue']
        plt.pie(stat,labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title('Feature Vectors')
        plt.show()

        # ploting a bar chart.
        y = [NubnerOfNon,numberOfZeros]
        N = len(y)
        width = 1/4
        barlist = plt.bar(labels, y, width)
        barlist[0].set_color('g')
        barlist[1].set_color('r')
        plt.show()


        from sklearn.metrics import classification_report, confusion_matrix
        # confusion matrix.
        print('Confusion matrix: ',confusion_matrix(testyforNow,y_pred))
        # Confusion report.
        print('Confusion report: ',classification_report(testyforNow,y_pred))
        # accuracy.
        print('Accuracy: ',svclassifier.score(testforNow, testyforNow))
        count1 =count1+count



