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
import pickle

# when read the data from CSV, the time stamp and associated GMC value should be reversed
def read_raw_data():
    x_1 =[]
    x_2 =[]

    for i in range (5):
        x_1=[]
        x_2=[]
        with open(r"./MealNoMealData/mealData"+str(i+1)+".csv",'rt')as f:
            data = csv.reader(f)
            for row in data:
                x_1.append(row)
        with open(r"./MealNoMealData/Nomeal"+str(i+1)+".csv",'rt')as ff:
            data = csv.reader(ff)
            for row in data:
                x_2.append(row)
        if i==0:
            x1 = x_1
            x2 = x_2
        elif i!=0:
            x1 = x1+x_1
            x2 = x2+x_2
    return x1,x2

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

x1,x2 = read_raw_data()
print('Number of rows from meal data:',len(x1))
print('Number of rows from no meal data:',len(x2))

x1 = smooth_data(x1)
print("Number of rows from the processed meal data: ",len(x1) )
x2 = smooth_data(x2)
print("Number of rows from the processed no meal data: ",len(x2))


# function for calculating the avg of changing velocity with window size 5, result in 6 features
def avg_vel(y):
    average = sum(y)/len(y)
    vel_y = []
    avg_vel = []
    window_size = 5
    for i in range (len(y)-1):
        vel = y[i+1]-y[i]
        vel_y.append(vel)
    np.asarray(vel_y)
#     xx = np.arange(29)
#     plt.plot(xx,vel_y)
#     plt.grid(True)
#     plt.show()
    for i in range (int(len(y)/window_size)):
        if i != (int((len(y)/window_size)-1)):
            avg = np.average(vel_y[(i*6):(i*6)+6])
        avg_vel.append(avg)
    avg_vel.append(average)
    array_vel = np.asarray(avg_vel)
    # array_vel = normalize(array_vel[:,np.newaxis], axis=0).ravel()
#     print('This is the avg_vel feature: ',array_vel)
    return array_vel


def FFT_feature(y):
    yf = 2.0/30 * np.abs(fft(y))
#     xf = np.linspace(0.0, 1.0, 15)
    yf = np.delete(yf,0)
    yf = np.unique(yf)
    xx = np.arange(15)
#     plt.plot(xx,yf[::-1])
#     plt.grid(True)
#     plt.show()
    max_yf = np.partition(yf,-6)[-6:]
    max_yf = np.asarray(max_yf)
    final_yf = normalize(max_yf[:,np.newaxis], axis=0).ravel()
#     print('This is the FFT feature: ',final_yf)
    return max_yf

# extract feature and save it into feature metricx
for i in range(len(x1)):
    yy = np.asarray(x1[i],dtype=np.float32)
    f1 = avg_vel(yy[:len(yy)-1])
    f2 = FFT_feature(yy)
    f12 = np.concatenate((f1, f2), axis=None)
    f1 = f12
#     f1 = FFT_feature(yy)
#     f1 = avg_vel(yy)
    if i == 0:
        feature_m1 = f1
    else:
        feature_m1 = np.vstack((feature_m1,f1))

for i in range(len(x2)):
    yy = np.asarray(x2[i],dtype=np.float32)
    f1 = avg_vel(yy[:len(yy)-1])
    f2 = FFT_feature(yy)
    f12 = np.concatenate((f1, f2), axis=None)
    f1 = f12
#     f1 = FFT_feature(yy)
#     f1 = avg_vel(yy)
    if i == 0:
        feature_m2 = f1
    else:
        feature_m2 = np.vstack((feature_m2,f1))


# merge two feature matrix together and create label set
label_1 = np.ones(len(x1))
label_2 = np.zeros(len(x2))
feature_m = np.vstack((feature_m1,feature_m2))
label = np.hstack((label_1,label_2))
print(label.shape)
print(feature_m.shape)


new_array = np.column_stack([feature_m,label])
print(new_array.shape)
print(feature_m.shape[1])
np.random.shuffle(new_array)
for i in range (len(new_array)):
    if i == 0:
        feature_m = new_array[i][:12]
        label = new_array[i][12]
    else:
        feature_m = np.vstack((feature_m,new_array[i][:12]))
        label = np.hstack((label, new_array[i][12]))


X_train = feature_m
y_train = label
clf = svm.SVC(kernel='poly', degree = 7,C=100).fit(X_train, y_train)
pickle.dump(clf, open('model.pickle', 'wb'))

kf = KFold(n_splits=5)
kf.get_n_splits(feature_m)
print("This is the K-fold cross validation result:")
for train_index, test_index in kf.split(feature_m):
    X_train, X_test = feature_m[train_index], feature_m[test_index]
    y_train, y_test = label[train_index], label[test_index]
    clf = svm.SVC(kernel='poly', degree = 7,C=100).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))