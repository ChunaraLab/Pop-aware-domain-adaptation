#Frustratingly easy domain adaptation model
#source datasets - 100 % train
#target data - 20% train, 80% test

import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_curve
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


TRAIN_DIRECTORY = "../../Data/"
TEST_DIRECTORY = "../../Sampled_data/"

common_symptoms = ['fever','cough','sorethroat','muscle','male','female','virus']

coefficients = defaultdict()

def read_file(filename):
    return pd.read_csv(filename)

def get_all_data(files_,common_symptoms):
    data = defaultdict()
    columns = defaultdict()
    dirs = ['Goviral','Hongkong','Hutterite']
    #get the source datasets
    for j,i in enumerate(files_):
        name = i
        name = name.replace('.csv','')
        temp = read_file(TEST_DIRECTORY+dirs[j]+"/Test/"+i)
        data[name] = temp[common_symptoms]
        columns[name] = list(data[name].columns)
        columns[name].remove('virus')

    return data,columns


def get_training_data(files_,common_symptoms,dir,name_):
    #first get all the source datasets
    #get the target dataset

    data = defaultdict()
    columns = defaultdict()
    #get the source datasets
    for i in files_:
        name = i
        name = name.replace('.csv','')
        temp = read_file(TRAIN_DIRECTORY+i)
        data[name] = temp[common_symptoms]
        columns[name] = list(data[name].columns)
        columns[name].remove('virus')
    
    #get the target datasets
    temp = read_file(TEST_DIRECTORY+dir+"/Train/"+name_)
    name_ = name_.replace('.csv','')
    data[name_] = temp[common_symptoms]
    columns[name_] = list(data[name_].columns)
    columns[name_].remove('virus')
    return data,columns



def overlap_columns(columns_):
    all_columns = list(columns_.values())
    overlap = list(set(all_columns[0]) & set(all_columns[1]) & set(all_columns[2]))
    return overlap
    


def create_columns(columns_):
    overlap = overlap_columns(columns_)
    new_columns = []
    temp = []
    for i in columns_.keys():
        x = [i.replace('.csv','')+'_'+j for j in columns_[i]]
        temp.append(x)
    t = [val for sublist in temp for val in sublist]
    new_columns = t + overlap
    new_columns.append('virus')
    return new_columns


def create_new_dataframe(data,columns):
    new_columns = create_columns(columns)
    new_dataset = defaultdict()
    for i,name in enumerate(data.keys()):
        new_data = pd.DataFrame(columns=new_columns)
        dataset = data[name]
        for j in columns[name]:
            new_data[name+'_'+j] = dataset[j]
            new_data[j] = dataset[j]
        new_data['virus'] = dataset['virus']
        new_data.fillna(0,inplace=True)
        new_dataset[name] = new_data
    #concatenate all the dataframe
    new_dataset = pd.concat(new_dataset.values())
    return new_dataset



def ml_model(dataset):
    lm = linear_model.LogisticRegression()
    x_train = dataset.drop(['virus'],axis = 1)
    y_train = dataset['virus']
    x = lm.fit(x_train,y_train)
    coeff = x.coef_.tolist()[0]
    return lm,coeff


#test_model
def test_model(train_data,test_data):
    lm,coeff = ml_model(train_data)
    train = test_data.drop(['virus'],axis = 1)
    test = test_data['virus']
    y_pred = lm.predict(train)
    acc = accuracy_score(test,y_pred)
    fpr,tpr,threshold = roc_curve(test,y_pred)
    auc_score = metrics.auc(fpr,tpr)
    return acc,auc_score
    

#create the test data
def create_data_for_testing(data,name,columns_):
    new_data = pd.DataFrame(columns = columns_)
    columns_for_data = list(data.columns)
    col = [x for x in columns_for_data if x != 'virus']
    for i in col:
        new_data[name+'_'+i] = data[i]
        new_data[i] = data[i]
    new_data['virus'] = data['virus']
    new_data.fillna(0,inplace = True)
    return new_data

def test_against_all(dataset_name,to_be_tested_names,data_,original_data):
    columns = list(original_data.columns)
    data = data_[dataset_name]
    for i in to_be_tested_names:
        temp_data = create_data_for_testing(data,i,columns)
        acc,auc_score = test_model(original_data,temp_data)
    return auc_score


# #### Heldout dataset : NYUMC
if __name__ == '__main__':

    files_ = ['goviral.csv','hongkong.csv','hutterite.csv']
    data_,columns = get_all_data(files_,common_symptoms)
    results = defaultdict()

    #Target dataset - Goviral
    files_goviral = ['hongkong.csv','hutterite.csv']
    data_goviral,columns_goviral = get_training_data(files_goviral,common_symptoms,'Goviral','goviral.csv')
    #create the new dataframe for domain adaptation
    new_dataset_goviral = create_new_dataframe(data_goviral,columns_goviral)
    coefficients['goviral'] = ml_model(new_dataset_goviral)

    print("Testing Goviral Data!\n")
    store_goviral = test_against_all('goviral',['goviral'],data_goviral,new_dataset_goviral)
    results['goviral'] = [store_goviral]

    #Target dataset - Hongkong
    files_hongkong = ['goviral.csv','hutterite.csv']
    data_hongkong,columns_hongkong = get_training_data(files_hongkong,common_symptoms,'Hongkong','hongkong.csv')
    #create the new dataframe for domain adaptation
    new_dataset_hongkong = create_new_dataframe(data_hongkong,columns_hongkong)
    coefficients['honkong'] = ml_model(new_dataset_hongkong)
    print("Testing Hongkong Data!\n")
    store_hongkong = test_against_all('hongkong',['hongkong'],data_hongkong,new_dataset_hongkong)
    results['hongkong'] = [store_hongkong]

    #Target dataset - Hutterite
    files_hutterite = ['goviral.csv','hongkong.csv']
    data_hutterite,columns_hutterite = get_training_data(files_hutterite,common_symptoms,'Hutterite','hutterite.csv')
    #create the new dataframe for domain adaptation
    new_dataset_hutterite = create_new_dataframe(data_hutterite,columns_hutterite)
    coefficients['hutterite'] = ml_model(new_dataset_hutterite)
    print("Testing Hutterite Data!\n")
    store_hutterite= test_against_all('hutterite',['hutterite'],data_hutterite,new_dataset_hutterite)
    results['hutterite'] = [store_hutterite]

    print(results)
    df = pd.DataFrame.from_records(results)
    df.to_csv("../Results/FEDA+p.csv",index=False)

