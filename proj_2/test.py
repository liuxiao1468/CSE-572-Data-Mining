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
import pickle


# Note that I assume you have one test data file for us to test
# You didn't clarify the format of the testing data with me
# I have to write it based on my assumption
# You have to modify the code if it doesn't fit with the testing data
# The file PATH MUST be modified

# when read the data from CSV, the time stamp and associated GMC value should be reversed
def read_raw_data():
    x_1 =[]
    with open(r"./MealNoMealData/mealData"+str(1)+".csv",'rt')as f:
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

x1= read_raw_data()
print('Number of rows from meal data:',len(x1))

x1 = smooth_data(x1)
print("Number of rows from the processed meal data: ",len(x1))


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
        feature_m = f1
    else:
        feature_m = np.vstack((feature_m,f1))

clf = pickle.load(open('model.pickle', 'rb'))
result = clf.predict(feature_m)
print(result)
np.savetxt('result.txt', result, fmt='%1.4e')
