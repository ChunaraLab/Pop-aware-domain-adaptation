#get the coeffficients to initialize the parameters of the hierarchical model
import warnings;
warnings.simplefilter('ignore')
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

DIRECTORY = "../../../../Data/"

def read_files(filename):
    data = pd.read_csv(filename)
    return data

symptoms = [
            'fever',
            'sorethroat',
            'cough',
            'muscle',
            ]
target_variable = 'virus'

#check if any column has null values
def nullColumns(dataset):
    null_columns = []
    for column in dataset.columns.tolist():
        if(dataset[column].isnull().sum() > 0):
            print(column)
            null_columns = null_columns.append(column)

#drop the rows with null values
def dropNull(dataset, null_list):
    dataset.dropna(axis=0,how='any', inplace = True)
    return dataset

def split_data(dataset):
    x = dataset[symptoms]
    y = dataset[target_variable]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 100)
    return x_train,x_test,y_train,y_test
    

def get_coeff(data):
    coefficients = []
    x_train,x_test,y_train,y_test = split_data(data)
    logistic_regression = linear_model.LogisticRegression()
    x  = logistic_regression.fit(x_train,y_train)
    y_pred = logistic_regression.predict(x_test)
    fpr,tpr,threshold = roc_curve(y_test,y_pred)
    auc_score = auc(fpr,tpr)
    print("AUC is :",auc_score)
    lr_intercept = x.intercept_
    coefficients.append(lr_intercept.tolist()[0])
    coeff = x.coef_.tolist()[0]
    for i in coeff:
        coefficients.append(i)
    return coefficients,auc_score


if __name__ == "__main__":

    coefficients = defaultdict()

    
    data_total = read_files(DIRECTORY+"total.csv")
    coeff_total,accuracy = get_coeff(data_total)
    print("_______________________________________")
    print("Coefficients : \n",coeff_total)
    coefficients['total'] = coeff_total

    data_age1 = read_files(DIRECTORY+"age1.csv")
    coeff_age1,accuracy = get_coeff(data_age1)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age1)
    coefficients['age1'] = coeff_age1

    data_age2 = read_files(DIRECTORY+"age2.csv")
    coeff_age2,accuracy = get_coeff(data_age2)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age2)
    coefficients['age2'] = coeff_age2

    data_age3 = read_files(DIRECTORY+"age3.csv")
    coeff_age3,accuracy = get_coeff(data_age3)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age3)
    coefficients['age3'] = coeff_age3

    data_age4 = read_files(DIRECTORY+"age4.csv")
    coeff_age4,accuracy = get_coeff(data_age4)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age4)
    coefficients['age4'] = coeff_age4

    data_age5 = read_files(DIRECTORY+"age5.csv")
    coeff_age5,accuracy = get_coeff(data_age5)
    print("_______________________________________")
    print("Coefficients : \n",coeff_age5)
    coefficients['age5'] = coeff_age5

    data_male = read_files(DIRECTORY+"male.csv")
    coeff_male,accuracy = get_coeff(data_male)
    print("_______________________________________")
    print("Coefficients : \n",coeff_male)
    coefficients['male'] = coeff_male

    data_female = read_files(DIRECTORY+"female.csv")
    coeff_female,accuracy = get_coeff(data_female)
    print("_______________________________________")
    print("Coefficients : \n",coeff_female)
    coefficients['female'] = coeff_female

    data_individually_reported = read_files(DIRECTORY+"individually_reported.csv")
    coeff_individually_reported,accuracy = get_coeff(data_individually_reported)
    print("_______________________________________")
    print("Coefficients : \n",coeff_individually_reported)
    coefficients['individually_reported'] = coeff_individually_reported

    data_health_worker_facilitated = read_files(DIRECTORY+"health_worker_facilitated.csv")
    coeff_health_worker_facilitated,accuracy = get_coeff(data_health_worker_facilitated)
    print("_______________________________________")
    print("Coefficients : \n",coeff_health_worker_facilitated)
    coefficients['health_worker_facilitated'] = coeff_health_worker_facilitated

    data_goviral = read_files(DIRECTORY+"goviral.csv")
    coeff_goviral,accuracy = get_coeff(data_goviral)
    print("_______________________________________")
    print("Coefficients : \n",coeff_goviral)
    coefficients['goviral'] = coeff_goviral

    data_fluwatch = read_files(DIRECTORY+"fluwatch.csv")
    coeff_fluwatch,accuracy = get_coeff(data_fluwatch)
    print("_______________________________________")
    print("Coefficients : \n",coeff_fluwatch)
    coefficients['fluwatch'] = coeff_fluwatch

    data_hongkong = read_files(DIRECTORY+"hongkong.csv")
    coeff_hongkong,accuracy = get_coeff(data_hongkong)
    print("_______________________________________")
    print("Coefficients : \n",coeff_hongkong)
    coefficients['hongkong'] = coeff_hongkong

    data_hutterite = read_files(DIRECTORY+"hutterite.csv")
    coeff_hutterite,accuracy = get_coeff(data_hutterite)
    print("_______________________________________")
    print("Coefficients : \n",coeff_hutterite)
    coefficients['hutterite'] = coeff_hutterite

    def roundup_decimals(list_):
        ans = [round(i,6) for i in list_]
        return ans

    temp = ['intercept'] + symptoms
    symptoms = temp
    results = pd.DataFrame.from_dict(coefficients)
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

    print(results)
    results.set_index('symptoms')
    results.to_csv("./Results/coefficients.csv",index = False)


