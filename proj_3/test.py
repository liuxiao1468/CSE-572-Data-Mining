#!/usr/bin/env python
# coding: utf-8

# In[33]:


# load data from csv file and save data into separate lists
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.fftpack import fft, ifft
from sklearn.decomposition import PCA
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
import pickle


# In[34]:


# Note that I assume you have one mealdata for us to test
# You didn't clarify the format of the testing data with me
# I have to write it based on my assumption
# You have to modify the code if it doesn't fit with the testing data

# when read the data from CSV, the time stamp and associated GMC value should be reversed
def read_raw_data():
    x_1 =[]
    with open(r'.\mealData'+str(1)+'.csv','rt')as f:# the path MUST be modified!!!!
        data = csv.reader(f)
        for row in data:
            x_1.append(row)
    return x_1

# this func is used to remove the data which contains 'NaN' and only use the first 30 data
def smooth_data(y):
    idx = []
    size_y = len(y)
    for i in range (size_y):
        y[i] = y[i][:30]
        y[i] = y[i][::-1]
        if (len(y[i])!= 30):
            idx.append(i)
        elif 'NaN' in y[i]:
            idx.append(i)      
    for j in range (len(idx),0,-1):
        del y[idx[j-1]]
    return y


# In[35]:


x1= read_raw_data()
print('Number of rows from meal data:',len(x1))

x1 = smooth_data(x1)
print("Number of rows from the processed meal data: ",len(x1))


# In[36]:


# function for calculating the avg of changing velocity with window size 3, result in 10 features
def avg_vel(y):
    average = sum(y)/len(y)
    vel_y = []
    avg_vel = []
    window_size = 3
    for i in range (len(y)-1):
        vel = y[i+1]-y[i]
        vel_y.append(vel)
    np.asarray(vel_y)

    for i in range (int(len(y)/window_size)):
        if i != (int((len(y)/window_size)-1)):
            avg = np.average(vel_y[(i*3):(i*3)+3])
        avg_vel.append(avg)
    array_vel = np.asarray(avg_vel)
    array_vel = normalize(array_vel[:,np.newaxis], axis=0).ravel()
#     array_vel = (array_vel - min(array_vel))/(max(array_vel)-min(array_vel))
    return array_vel

# function for calculating the avg of meal amount with window size 3, result in 10 features
def avg_win(y):
    avg_win = []
    window_size = 3
    for i in range (int(len(y)/window_size)):
        if i != (int((len(y)/window_size)-1)):
            avg = np.average(y[(i*3):(i*3)+3])
        avg_win.append(avg)
    array_win = np.asarray(avg_win)
    array_win = normalize(array_win[:,np.newaxis], axis=0).ravel()
#     array_vel = (array_vel - min(array_vel))/(max(array_vel)-min(array_vel))
    return array_win

def max_increase(y):
    change = []
    y = list(map(int, y))
    y_0 = y[5]
    y_max = max(y[5:])
    y_end = y[29]
    max_increase = (y_max - y_0)/y_0
    max_decrease = (y_max - y_end)/y_end
    before_change = max(y[:5])-min(y[:5])
    change.append(max_increase)
    change.append(max_decrease)
    change.append(before_change)
    change = np.asarray(change,dtype=np.float32)
    changed = normalize(change[:,np.newaxis], axis=0).ravel()
#     changed = (change-min(change))/(max(change)-min(change))
    return changed


# In[37]:


# extract feature and save it into feature metricx
for i in range(len(x1)):
    yy = np.asarray(x1[i],dtype=np.float32)
    f1 = avg_vel(yy)
    f2 = max_increase(yy)
    f1 = np.concatenate((f1, f2), axis=None)
#     f3 = avg_win(yy)
#     f1 = np.concatenate((f12, f3), axis=None)
    if i == 0:
        feature_m1 = f1
    else:
        feature_m1 = np.vstack((feature_m1,f1))


# In[38]:


feature = open("feature_m1.pkl","rb")
feature = pickle.load(feature)
k_means_label = open("kmeans_label.pkl","rb")
k_means_label = pickle.load(k_means_label)
dbscan_label = open("dbscan_label.pkl","rb")
dbscan_label = pickle.load(dbscan_label)


# In[39]:


knn_kmeans = KNeighborsClassifier(n_neighbors=20)
knn_kmeans.fit(feature, k_means_label)
y_predict1 = knn_kmeans.predict(feature_m1)
y_predict1 = np.asarray(y_predict1)
y_predict1 = y_predict1.transpose()


# In[40]:


knn_dbscan = KNeighborsClassifier(n_neighbors=20)
knn_dbscan.fit(feature, dbscan_label)
y_predict2 = knn_dbscan.predict(feature_m1)
y_predict2 = np.asarray(y_predict2)
y_predict2 = y_predict2.transpose()


# In[41]:


new_array = np.column_stack([y_predict1,y_predict2])


# In[42]:


np.savetxt('result.csv', new_array, fmt="%d", delimiter=",")
print("------ save result to result.csv ------")


# In[ ]:




