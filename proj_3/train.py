#!/usr/bin/env python
# coding: utf-8

# In[532]:


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


# In[533]:


# when read the data from CSV, the time stamp and associated GMC value should be reversed
def read_raw_data():
    x_1 =[]
    x_2 =[]

    for i in range (5):
        x_1=[]
        x_2=[]
        with open(r'.\mealData'+str(i+1)+'.csv','rt')as f:
            data = csv.reader(f)
            rows_x=[row for idx, row in enumerate(data) if idx<50]# only use first 20 rows of the data
            for row in rows_x:
                x_1.append(row)
        with open(r'.\MealAmountData'+str(i+1)+'.csv','rt')as ff:
            data = csv.reader(ff)
            rows_x=[row for idx, row in enumerate(data) if idx<50]# only use first 20 rows of the data
            for row in rows_x:
                x_2.append(row)
        if i==0:
            x1 = x_1
            x2 = x_2
        elif i!=0:
            x1 = x1+x_1
            x2 = x2+x_2
    return x1,x2

# this func is used to remove the data which contains 'NaN' and only use the first 30 data
def smooth_data(y,x):
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
        del x[idx[j-1]]
    return y, x


# In[534]:


x1,x2 = read_raw_data()
print('Number of rows from meal data:',len(x1))
print('Number of meal amount data:',len(x2))

x1, x2 = smooth_data(x1, x2)
print("Number of rows from the processed meal data: ",len(x1) )
print("Number of rows from the processed meal amount data: ",len(x2))
print()


# In[535]:


def extract_ground_truth(x2):
    bin_truth = []
    for i in range (len(x2)):
        if int(x2[i][0]) == 0:
            bin_truth.append(1)
        elif (int(x2[i][0])>0) and (int(x2[i][0])<=20):
            bin_truth.append(2)
        elif (int(x2[i][0])>20) and (int(x2[i][0])<=40):
            bin_truth.append(3)
        elif (int(x2[i][0])>40) and (int(x2[i][0])<=60):
            bin_truth.append(4)
        elif (int(x2[i][0])>60) and (int(x2[i][0])<=80):
            bin_truth.append(5)
        elif (int(x2[i][0])>80) and (int(x2[i][0])<=100):
            bin_truth.append(6)
    return bin_truth


# In[536]:


bin_truth = extract_ground_truth(x2)
bin_truth = np.asarray(bin_truth)
# print("number of points in Bin 1",bin_truth.count(1))
# print("number of points in Bin 2",bin_truth.count(2))
# print("number of points in Bin 3",bin_truth.count(3))
# print("number of points in Bin 4",bin_truth.count(4))
# print("number of points in Bin 5",bin_truth.count(5))
# print("number of points in Bin 6",bin_truth.count(6))


# In[537]:


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


# In[546]:


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
with open('feature_m1.pkl','wb') as f:
    pickle.dump(feature_m1, f)


# In[547]:


def try_model_kmeans(label,feature_m1,n_cluster,idx_keep,bin_truth):
    cluster = []
    bin_cluster = []
    bin_index = []
    result = []
    bin_1 = []
    bin_2 = []
    bin_3 = []
    bin_4 = []
    bin_5 = []
    bin_6 = []
    bin_1_idx = []
    bin_2_idx = []
    bin_3_idx = []
    bin_4_idx = []
    bin_5_idx = []
    bin_6_idx = []
    idx_save = []
    for j in range (n_cluster):
        cluster_j = [ i for i in range(len(label)) if label[i] == j ]
        idx_save_j = [idx_keep[i] for i in cluster_j]
                
        cluster.append(cluster_j)
        idx_save.append(idx_save_j)
    for j in range (n_cluster):
        result_label = [bin_truth[i] for i in cluster[j]]
#         print(max(set(result_label), key=result_label.count), " ", len(result_label))
        result.append(max(set(result_label), key=result_label.count))
    for k in range (n_cluster):
        if result[k]==1:
            bin_1 = bin_1 + cluster[k]
            bin_1_idx = bin_1_idx + idx_save[k]
        elif result[k]==2:
            bin_2 = bin_2 + cluster[k]
            bin_2_idx = bin_2_idx + idx_save[k]
        elif result[k]==3:
            bin_3 = bin_3 + cluster[k]
            bin_3_idx = bin_3_idx + idx_save[k]
        elif result[k]==4:
            bin_4 = bin_4 + cluster[k]
            bin_4_idx = bin_4_idx + idx_save[k]
        elif result[k]==5:
            bin_5 = bin_5 + cluster[k]
            bin_5_idx = bin_5_idx + idx_save[k]
        elif result[k]==6:
            bin_6 = bin_6 + cluster[k]
            bin_6_idx = bin_6_idx + idx_save[k]
    bin_cluster.append(bin_1)
    bin_cluster.append(bin_2)
    bin_cluster.append(bin_3)
    bin_cluster.append(bin_4)
    bin_cluster.append(bin_5)
    bin_cluster.append(bin_6)
    bin_index.append(bin_1_idx)
    bin_index.append(bin_2_idx)
    bin_index.append(bin_3_idx)
    bin_index.append(bin_4_idx)
    bin_index.append(bin_5_idx)
    bin_index.append(bin_6_idx)
    return bin_cluster, bin_index

def accuracy_report(feature_m1,bin_cluster,bin_index,bin_truth):
    final_result = [0]*len(feature_m1)

    final_bin1 = bin_index[0]
    final_bin2 = bin_index[1]
    final_bin3 = bin_index[2]
    final_bin4 = bin_index[3]
    final_bin5 = bin_index[4]
    final_bin6 = bin_index[5]

    label_bin1 = [1]*len(final_bin1)
    label_bin2 = [2]*len(final_bin2)
    label_bin3 = [3]*len(final_bin3)
    label_bin4 = [4]*len(final_bin4)
    label_bin5 = [5]*len(final_bin5)
    label_bin6 = [6]*len(final_bin6)
    

    for (i, j) in zip(final_bin1, label_bin1):
        final_result[i] = j

    for (i, j) in zip(final_bin2, label_bin2):
        final_result[i] = j

    for (i, j) in zip(final_bin3, label_bin3):
        final_result[i] = j

    for (i, j) in zip(final_bin4, label_bin4):
        final_result[i] = j

    for (i, j) in zip(final_bin5, label_bin5):
        final_result[i] = j

    for (i, j) in zip(final_bin6, label_bin6):
        final_result[i] = j
    
    score = 0
    for i in range(len(final_result)):
        if (final_result[i] == bin_truth[i]):
            score = score +1
    final_score = score/(len(final_result))
#     print("k_means accuracy is: ", final_score)
    return final_result

def knn_score(y_predict, y_test):
    score = 0
    for i in range(len(y_predict)):
        if (y_predict[i] == y_test[i]):
            score = score+1
    final_score = score/(len(y_predict))
    print("KNN accuracy ------> ", final_score)


# In[548]:


kf = KFold(n_splits=5)
kf.get_n_splits(feature_m1)
print("This is the K-fold cross validation result:")
print("--------------------------------------")
print()
ii = 0
for train_index, test_index in kf.split(feature_m1):
    X_train, X_test = feature_m1[train_index], feature_m1[test_index]
    y_train, y_test = bin_truth[train_index], bin_truth[test_index]
    kmeans = KMeans(n_clusters= 85, random_state=0).fit(X_train)
    print("k-fold when k =",ii+1)
    print("SSE value of the clusters: ",kmeans.inertia_)
    label = kmeans.labels_
    label = list(label)
    index_keep = [i for i in range (len(label))]
    bin_cluster, bin_index = try_model_kmeans(label,X_train,85,index_keep,y_train)
#     print("Kmeans result:")
#     print("Bin","Num")
#     for i in range (6):
#         print(i+1, " ", len(bin_cluster[i]))
    final_label = accuracy_report(X_train,bin_cluster,bin_index,y_train)
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, final_label)
    y_predict = knn.predict(X_test)
    knn_score(y_predict,y_test)
    ii = ii+1
    print()


# In[549]:


print("---Train the cluster by Kmeans with full dataset----")
kmeans = KMeans(n_clusters= 85 , random_state=0).fit(feature_m1)
print("SSE value of the clusters: ",kmeans.inertia_)
label = kmeans.labels_
label = list(label)
index_keep = [i for i in range (len(label))]
bin_cluster, bin_index = try_model_kmeans(label,feature_m1,85,index_keep,bin_truth)
final_label = accuracy_report(feature_m1,bin_cluster,bin_index,bin_truth)
with open('kmeans_label.pkl','wb') as f:
    pickle.dump(final_label, f)
print("---Save the result into a pickle file---")
print()


# In[550]:


def try_model_dbscan(label,feature_m1,label_class,idx_keep,bin_truth):
    cluster = []
    bin_cluster = []
    bin_index = []
    result = []
    bin_1 = []
    bin_2 = []
    bin_3 = []
    bin_4 = []
    bin_5 = []
    bin_6 = []
    bin_1_idx = []
    bin_2_idx = []
    bin_3_idx = []
    bin_4_idx = []
    bin_5_idx = []
    bin_6_idx = []
    idx_save = []
    for j in range (len(label_class)):
        cluster_j = [ i for i in range(len(label)) if label[i] == label_class[j] ]
        idx_save_j = [idx_keep[i] for i in cluster_j]
                
        cluster.append(cluster_j)
        idx_save.append(idx_save_j)
    for j in range (len(label_class)):
        result_label = [bin_truth[i] for i in cluster[j]]
        result.append(max(set(result_label), key=result_label.count))
    for k in range (len(label_class)):
        if result[k]==1:
            bin_1 = bin_1 + cluster[k]
            bin_1_idx = bin_1_idx + idx_save[k]
        elif result[k]==2:
            bin_2 = bin_2 + cluster[k]
            bin_2_idx = bin_2_idx + idx_save[k]
        elif result[k]==3:
            bin_3 = bin_3 + cluster[k]
            bin_3_idx = bin_3_idx + idx_save[k]
        elif result[k]==4:
            bin_4 = bin_4 + cluster[k]
            bin_4_idx = bin_4_idx + idx_save[k]
        elif result[k]==5:
            bin_5 = bin_5 + cluster[k]
            bin_5_idx = bin_5_idx + idx_save[k]
        elif result[k]==6:
            bin_6 = bin_6 + cluster[k]
            bin_6_idx = bin_6_idx + idx_save[k]
    bin_cluster.append(bin_1)
    bin_cluster.append(bin_2)
    bin_cluster.append(bin_3)
    bin_cluster.append(bin_4)
    bin_cluster.append(bin_5)
    bin_cluster.append(bin_6)
    bin_index.append(bin_1_idx)
    bin_index.append(bin_2_idx)
    bin_index.append(bin_3_idx)
    bin_index.append(bin_4_idx)
    bin_index.append(bin_5_idx)
    bin_index.append(bin_6_idx)
    return bin_cluster, bin_index

def accuracy_report_dbscan(feature_m1,bin_cluster,bin_index,bin_truth):
    temp = []
    for i in range (6):
        temp.append(len(bin_cluster[i]))
    aa = temp.index(max(temp))  
    f1 = [feature_m1[i] for i in bin_cluster[aa]]
    bin_index_1 = bin_index[aa]
    kmeans_1 = KMeans(n_clusters=60, random_state=0).fit(f1)
    label_1 = list(kmeans_1.labels_)
    bin_cluster_2nd, bin_index_2nd = try_model_kmeans(label_1,f1,60,bin_index_1,bin_truth)
#     print("2-nd Kmeans result:")
#     print("Bin","Num")
#     for i in range (6):
#         print(i+1, " ", len(bin_cluster_2nd[i]))
#     print()
    final_result = [0]*len(feature_m1)
    final_bin1 = bin_index_2nd[0]
    final_bin2 = bin_index[1]+bin_index_2nd[1]
    final_bin3 = bin_index[2]+bin_index_2nd[2]
    final_bin4 = bin_index[3]+bin_index_2nd[3]
    final_bin5 = bin_index[4]+bin_index_2nd[4]
    final_bin6 = bin_index[5]+bin_index_2nd[5]

#     final_bin1 = bin_index[0]
#     final_bin2 = bin_index[1]
#     final_bin3 = bin_index[2]
#     final_bin4 = bin_index[3]
#     final_bin5 = bin_index[4]
#     final_bin6 = bin_index[5]

    label_bin1 = [1]*len(final_bin1)
    label_bin2 = [2]*len(final_bin2)
    label_bin3 = [3]*len(final_bin3)
    label_bin4 = [4]*len(final_bin4)
    label_bin5 = [5]*len(final_bin5)
    label_bin6 = [6]*len(final_bin6)
    

    for (i, j) in zip(final_bin1, label_bin1):
        final_result[i] = j

    for (i, j) in zip(final_bin2, label_bin2):
        final_result[i] = j

    for (i, j) in zip(final_bin3, label_bin3):
        final_result[i] = j

    for (i, j) in zip(final_bin4, label_bin4):
        final_result[i] = j

    for (i, j) in zip(final_bin5, label_bin5):
        final_result[i] = j

    for (i, j) in zip(final_bin6, label_bin6):
        final_result[i] = j
    
    score = 0
    for i in range(len(final_result)):
        if (final_result[i] == bin_truth[i]):
            score = score +1
    final_score = score/(len(final_result))
    print("SSE value of the clusters: ",kmeans_1.inertia_)
#     print("reported accuracy is: ", final_score)
    return final_result


# In[551]:


kf = KFold(n_splits=5)
kf.get_n_splits(feature_m1)
print("This is the K-fold cross validation result on DBSCAN method:")
print("--------------------------------------")
print()

ii = 0
for train_index, test_index in kf.split(feature_m1):
    X_train, X_test = feature_m1[train_index], feature_m1[test_index]
    y_train, y_test = bin_truth[train_index], bin_truth[test_index]
    dbscan = DBSCAN(eps=0.5, min_samples=3).fit(X_train)
    label = dbscan.labels_
    label_class = np.unique(label)
    label = list(label)
    index_keep = [i for i in range (len(label))]
    bin_cluster, bin_index = try_model_dbscan(label,X_train,label_class,index_keep,y_train)
#     print("1-st dbscan result:")
#     print("Bin","Num")
#     for i in range (6):
#         print(i+1, " ", len(bin_cluster[i]))
    final_label = accuracy_report_dbscan(X_train,bin_cluster,bin_index,y_train)
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, final_label)
    y_predict = knn.predict(X_test)
    knn_score(y_predict,y_test)
    ii = ii+1
    print()


# In[552]:


print("---Train the cluster using DBSCAN with full dataset----")
dbscan = DBSCAN(eps=0.5, min_samples=3).fit(feature_m1)
label = dbscan.labels_
label_class = np.unique(label)
# print(label)
# print(label_class)
label = list(label)
index_keep = [i for i in range (len(label))]
bin_cluster, bin_index = try_model_dbscan(label,feature_m1,label_class,index_keep,bin_truth)
final_label = accuracy_report_dbscan(feature_m1,bin_cluster,bin_index,bin_truth)
with open('dbscan_label.pkl','wb') as f:
    pickle.dump(final_label, f)
print("---Save the result into a pickle file---")


# In[ ]:




