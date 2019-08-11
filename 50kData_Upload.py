'''
Dos attack detection using svm clasifier.
Dataset contains 656580 records.
Approah Train/Test Split.
'''
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
#matplotlib inline.

# attack free data 1.
Attackfree1 = pd.read_csv("C:/Users/Lovleen kaur/Desktop/Summer Project/Attacks/Attackfree csv files/Attackfree1.csv")
# removing useless columns.
Attackfree1 = Attackfree1[['Timestamp','ID','DLC','Message']]
# droping NAN values.
Attackfree1 = Attackfree1.dropna()
# replacing hex value with decimal values for ID column.
Attackfree1['ID'] = Attackfree1['ID'].apply(lambda x: x if str(x).isdigit() else int(str(x),16))
# replacing hex value with decimal values for Message column.
Attackfree1['Message'] = Attackfree1['Message'].apply(lambda x: x if str(x).isdigit() else int(str(x),16))
#print(Attackfree1[1:100])

#dos data 2.
dos = pd.read_csv("C:/Users/Lovleen kaur/Desktop/Summer Project/Attacks/dos.csv")
# Removing use less column.
dos = dos[['Timestamp','ID','DLC','Message']]
# Droping NAN values.
dos = dos.dropna()
# replacing hex value with decimal values for ID column.
dos['ID'] = dos['ID'].apply(lambda x: x if str(x).isdigit() else int(str(x),16))
# replacing hex value with decimal values for Message column.
dos['Message'] = dos['Message'].apply(lambda x: x if str(x).isdigit() else int(str(x),16))
#print(dos[1:100])

frames = [Attackfree1[0:40000],dos[0:10000]]
# merging all three dataframes into one.
finalDataset = pd.concat(frames)
print(finalDataset)
# importing train_test_split method.
from sklearn.model_selection import train_test_split
# X contains all columns except ID.
X = finalDataset.drop('ID', axis=1)
#Contains only ID columns.
y = finalDataset['ID']
y=y.astype('int')
# Spliting the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,  shuffle=True)
# importing support vector classifier.
print("done")
