import pandas as pd
import numpy as np
import os
import time
from sklearn import linear_model
from sklearn.metrics import f1_score,roc_curve,precision_score,recall_score,accuracy_score,auc
from collections import defaultdict
from sklearn import metrics
import random
random.seed(10)


#Only target dataset
#20% - training , 80% - test


symptoms = ['fever','cough','muscle','sorethroat','virus']


def read_files(filename):
    data = pd.read_csv(filename)
    data = data[symptoms]
    return data


def get_training_data(train_files_,train_directory):
    data = []
    for i in train_files_:
        print(train_directory+i)
        data.append(read_files(train_directory+i))
        
    training_data = pd.concat(data)
    x_train = training_data.drop(['virus'],axis = 1)
    y_train = training_data['virus']
    return x_train,y_train

def get_test_data(test_file_,test_directory):
    data = read_files(test_directory+test_file_)
    x_test = data.drop(['virus'],axis = 1)
    y_test = data['virus']
    return x_test,y_test

def linear_regression_model(x_train,x_test,y_train,y_test):
    lm = linear_model.LogisticRegression()
    lm.fit(x_train,y_train)
    y_pred = lm.predict(x_test)
    fpr,tpr,threshold = roc_curve(y_test,y_pred)
    auc_score = auc(fpr,tpr)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    return auc_score,accuracy,precision,recall,f1


def predict_on_unknown(train_files,test_file,train_directory,test_directory ):
    results = defaultdict()
    x_train,y_train = get_training_data(train_files,train_directory)
    x_test,y_test = get_test_data(test_file,test_directory)
    auc_score,accuracy,precision,recall,f1 = linear_regression_model(x_train,x_test,y_train,y_test)
    results['AUC'] = auc_score
    results['Accuracy'] = accuracy
    results['Precision'] = precision
    results['Recall'] = recall
    results['F1'] = f1
    for k,v in results.items():
        print(k,"\t",v)
    return results


# ### Predict on GoViral


TRAIN_DIRECTORY = "../../Sampled_data/Goviral/Train/"
TEST_DIRECTORY = "../../Sampled_data/Goviral/Test/"




# train_files = ['nyumc.csv','goviral.csv','fluwatch.csv','hongkong.csv','hutterite.csv']
train_files = ['goviral.csv']
test_file = 'goviral.csv'


print("GoViral!")
results_goviral = predict_on_unknown(train_files,test_file,TRAIN_DIRECTORY,TEST_DIRECTORY)


# ### Predict on FluWatch

TRAIN_DIRECTORY = "../../Sampled_data/Fluwatch/Train/"
TEST_DIRECTORY = "../../Sampled_data/Fluwatch/Test/"


# train_files = ['nyumc.csv','goviral.csv','fluwatch.csv','hongkong.csv','hutterite.csv']
train_files = ['fluwatch.csv']
test_file = 'fluwatch.csv'
print("FluWatch!")
results_fluwatch = predict_on_unknown(train_files,test_file,TRAIN_DIRECTORY,TEST_DIRECTORY)


# ### Predict on HongKong


TRAIN_DIRECTORY = "../../Sampled_data/Hongkong/Train/"
TEST_DIRECTORY = "../../Sampled_data/Hongkong/Test/"


# train_files = ['nyumc.csv','goviral.csv','fluwatch.csv','hongkong.csv','hutterite.csv']
train_files = ['hongkong.csv']
test_file = 'hongkong.csv'
print("HongKong")
results_hongkong = predict_on_unknown(train_files,test_file,TRAIN_DIRECTORY ,TEST_DIRECTORY)


# ### Predict on Hutterite


TRAIN_DIRECTORY = "../../Sampled_data/Hutterite/Train/"
TEST_DIRECTORY = "../../Sampled_data/Hutterite/Test/"


# train_files = ['nyumc.csv','goviral.csv','fluwatch.csv','hongkong.csv','hutterite.csv']
train_files = ['hutterite.csv']
test_file = 'hutterite.csv'
print("Hutterite!")
results_hutterite = predict_on_unknown(train_files,test_file,TRAIN_DIRECTORY,TEST_DIRECTORY)


# ### Predict on Loeb


TRAIN_DIRECTORY = "../Data/Symptoms_Demo/Loeb/Train/"
TEST_DIRECTORY = "../Data/Symptoms_Demo/Loeb/Test/"



# train_files = ['nyumc.csv','goviral.csv','fluwatch.csv','hongkong.csv','hutterite.csv']
train_files = ['loeb.csv']
test_file = 'loeb.csv'
print("Loeb!")
results_hutterite = predict_on_unknown(train_files,test_file,TRAIN_DIRECTORY,TEST_DIRECTORY)

