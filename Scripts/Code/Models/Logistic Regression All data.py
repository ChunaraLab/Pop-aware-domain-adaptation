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


# One dataset is held out. All the other datasets are concatenated. 
# We implement a Logistic Regression on the concatenated data and then test out on the held out the data.



symptoms = ['fever','cough','muscle','sorethroat','virus']


#read the csv file and only consider the symptoms column features
def read_files(filename):
    data = pd.read_csv(filename)
    data = data[symptoms]
    return data


#get the training data
#this is done in two steps
#first get 100% of all the source datasets
#second get 20% of the target datasets
def get_training_data(train_files_,train_directory,test_file_,test_directory):
    data = []

    #get the source datasets
    for i in train_files_:
        data.append(read_files(train_directory+i))
    

    #get the target dataset
    target_train_dir = test_directory+"Train/"
    data.append(read_files(target_train_dir+test_file_))

    #just concatenate all the training data to get a single dataset
    training_data = pd.concat(data)
    x_train = training_data.drop(['virus'],axis = 1)
    y_train = training_data['virus']
    return x_train,y_train



def get_test_data(test_file_,test_directory):
    test_directory = test_directory+"Test/"
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
    x_train,y_train = get_training_data(train_files,train_directory,test_file,test_directory)
    x_test,y_test = get_test_data(test_file,test_directory)
    auc_score,accuracy,precision,recall,f1 = linear_regression_model(x_train,x_test,y_train,y_test)
    results['AUC'] = auc_score
    results['Accuracy'] = accuracy
    results['Precision'] = precision
    results['Recall'] = recall
    results['F1'] = f1
    # for k,v in results.items():
    #     print(k,"\t",v)
    return results

if __name__ == '__main__':

    TRAIN_DIRECTORY = "../../Data/"
    TEST_DIRECTORY = "../../Sampled_Data/"

    #training data consists of all the source datasets as well as 20% of the target dataset
    
    # Goviral target dataset
    train_files = ['hongkong.csv','hutterite.csv']
    test_file = 'goviral.csv'
    print("\nAnalyzing for Goviral dataset!")
    result_goviral = predict_on_unknown(train_files,test_file,TRAIN_DIRECTORY,TEST_DIRECTORY+"Goviral/")

    #Hongkong target dataset
    train_files = ['goviral.csv','hutterite.csv']
    test_file = 'hongkong.csv'
    print("\nAnalyzing Hongkong dataset!")
    result_hongkong = predict_on_unknown(train_files,test_file,TRAIN_DIRECTORY,TEST_DIRECTORY+"Hongkong/")

    #Hutterite target dataset
    train_files = ['goviral.csv','hongkong.csv']
    test_file = 'hutterite.csv'
    print("\nAnalyzing hutterite dataset!")
    result_hutterite = predict_on_unknown(train_files,test_file,TRAIN_DIRECTORY,TEST_DIRECTORY+"Hutterite/")

    results_logistic_regression = defaultdict()
    results_logistic_regression['Goviral'] = [result_goviral['AUC']]
    results_logistic_regression['Hongkong'] = [result_hongkong['AUC']]
    results_logistic_regression['Hutterite'] = [result_hutterite['AUC']]

    print(results_logistic_regression)
    df = pd.DataFrame.from_records(results_logistic_regression)
    df.to_csv("../Results/logistic_regression_flat.csv",index=False)


