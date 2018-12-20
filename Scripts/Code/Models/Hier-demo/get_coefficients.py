#get the initial parameters for the hierarchical model
#uses logisitic regression to get these parameters
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import time
import os
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm
from collections import defaultdict
from sklearn.metrics import accuracy_score,roc_curve,auc

#the following symptoms are considered in the studt
symptoms = [
            'fever',
            'sorethroat',
            'cough',
            'muscle',
            ]
target_variable = 'virus'


def read_files(filename):
    data = pd.read_csv(filename)
    return data

#data directory : has the processed data for four datasets -
#goviral, fluwatch, hongkong, hutterite

directory = "../../../Data/"
#data directory : has the processed data for four datasets as well as
#the data for the levels in the hierarchy
#goviral, fluwatch, hongkong, hutterite


#drop the rows with null values
def dropNull(dataset):
    dataset.dropna(axis=0,how='any', inplace = True)
    return dataset


def split_data(dataset):
    x = dataset[symptoms]
    y = dataset[target_variable]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1, random_state = 100)
    return x_train,x_test,y_train,y_test

def get_coeff(data):
    coefficients_1 = []
    x_train,x_test,y_train,y_test = split_data(data)
#     print("Training data is :\n",x_train.head())
    logistic_regression = linear_model.LogisticRegression()
    x  = logistic_regression.fit(x_train,y_train)
    y_pred = logistic_regression.predict(x_test)
#     print("Predicted values are : ",y_pred)
    fpr,tpr,threshold = roc_curve(y_test,y_pred)
    auc_score = auc(fpr,tpr)
    accuracy = accuracy_score(y_test,y_pred)
    print("AUC is :",auc_score)
    lr_intercept = x.intercept_
    coefficients_1.append(lr_intercept.tolist()[0])
    coeff = x.coef_.tolist()[0]
    for i in coeff:
        coefficients_1.append(i)
    return coefficients_1,auc_score


def roundup_decimals(list_):
    ans = [round(i,6) for i in list_]
    return ans

def create_coeff_dataframe(coefficients):
    #create the dataframe for  storing the coefficients
    results = pd.DataFrame.from_dict(coefficients)
    symptoms.insert(0,'intercept')
    results['symptoms'] = symptoms

    cols = list(results.columns)
    cols = cols[-1:] + cols[:-1]
    results = results[cols]


    results['symptoms'] = symptoms
    results['goviral'] = coeff_goviral
    results['fluwatch'] = coeff_fluwatch
    results['hongkong'] = coeff_hongkong
    results['hutterite'] = coeff_hutterite
    results['age 0-4'] = coeff_age1
    results['age 5-15'] = coeff_age2
    results['age 16-44'] = coeff_age3
    results['age 45-64'] = coeff_age4
    results['age 65+'] = coeff_age5
    results['male'] = coeff_male
    results['female'] = coeff_female
    results['individually collected'] = coeff_individually_reported
    results['health worker facilitated'] = coeff_health_worker_facilitated
    results['total'] =coeff_total

    results.head()
    results.set_index('symptoms',inplace = True)
    # results.to_csv("./Results/coefficients.csv",index = False)
    return results

if __name__ == "__main__":

    #coefficients holds the logistic regression coefficients for the different data groups
    coefficients = defaultdict()

    print("Getting the coefficients for all the data!")
    #get the parameters across all the datasets
    data_total = read_files(directory+"total.csv")
    print("Read the data!")
    coeff_total,accuracy = get_coeff(data_total)
    print("_______________________________________")
    print("Coefficients : \n",coeff_total)
    coefficients['total'] = coeff_total

    #get the parameters for the age group 0-5
    data_age1 = read_files(directory+"age1.csv")
    coeff_age1,accuracy = get_coeff(data_age1)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age1)
    coefficients['age1'] = coeff_age1

    #get the parameters for the age group 5-15
    data_age2 = read_files(directory+"age2.csv")
    coeff_age2,accuracy = get_coeff(data_age2)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age2)
    coefficients['age2'] = coeff_age2

    #get the parameters for the age group 16-44
    data_age3 = read_files(directory+"age3.csv")
    coeff_age3,accuracy = get_coeff(data_age3)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age3)
    coefficients['age3'] = coeff_age3

    #get the parameters for the age group 45-64
    data_age4 = read_files(directory+"age4.csv")
    coeff_age4,accuracy = get_coeff(data_age4)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age4)
    coefficients['age4'] = coeff_age4

    #get the parameters for the age group 65+
    data_age5 = read_files(directory+"age5.csv")
    coeff_age5,accuracy = get_coeff(data_age5)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age5)
    coefficients['age5'] = coeff_age5

    #get the parameters for males
    data_male = read_files(directory+"male.csv")
    coeff_male,accuracy = get_coeff(data_male)
    print("_______________________________________")
    print("Coefficients : \n",coeff_male)
    coefficients['male'] = coeff_male

    #get the parameters for femals
    data_female = read_files(directory+"female.csv")
    coeff_female,accuracy = get_coeff(data_female)
    print("_______________________________________")
    print("Coefficients : \n",coeff_female)
    coefficients['female'] = coeff_female

    #get the parameters for the individually_reported group
    data_individually_reported = read_files(directory+"individually_reported.csv")
    coeff_individually_reported,accuracy = get_coeff(data_individually_reported)
    print("_______________________________________")
    print("Coefficients : \n",coeff_individually_reported)
    coefficients['individually_reported'] = coeff_individually_reported

    #get the parameters for the health worker facilitated groups
    data_health_worker_facilitated = read_files(directory+"health_worker_facilitated.csv")
    coeff_health_worker_facilitated,accuracy = get_coeff(data_health_worker_facilitated)
    print("_______________________________________")
    print("Coefficients : \n",coeff_health_worker_facilitated)
    coefficients['health_worker_facilitated'] = coeff_health_worker_facilitated

    #get the parameters for the goviral dataset
    data_goviral = read_files(directory+"goviral.csv")
    coeff_goviral,accuracy = get_coeff(data_goviral)
    print("_______________________________________")
    print("Coefficients : \n",coeff_goviral)
    coefficients['goviral'] = coeff_goviral

    #get the parameters for the fluwatch dataset
    data_fluwatch = read_files(directory+"fluwatch.csv")
    coeff_fluwatch,accuracy = get_coeff(data_fluwatch)
    print("_______________________________________")
    print("Coefficients : \n",coeff_fluwatch)
    coefficients['fluwatch'] = coeff_fluwatch

    #get the parameters for the hongkong dataset
    data_hongkong = read_files(directory+"hongkong.csv")
    coeff_hongkong,accuracy = get_coeff(data_hongkong)
    print("_______________________________________")
    print("Coefficients : \n",coeff_hongkong)
    coefficients['hongkong'] = coeff_hongkong

    #get the parameters for the hutterite dataset
    data_hutterite = read_files(directory+"hutterite.csv")
    coeff_hutterite,accuracy = get_coeff(data_hutterite)
    print("_______________________________________")
    print("Coefficients : \n",coeff_hutterite)
    coefficients['hutterite'] = coeff_hutterite

    results = create_coeff_dataframe(coefficients)
    print(results)

    print("Saving the coefficients!")
    results.to_csv("./Coefficients/LR_coefficients.csv")
