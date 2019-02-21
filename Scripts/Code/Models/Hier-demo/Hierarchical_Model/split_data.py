import pandas as pd
import numpy as np
from data_preprocessing import *
import os


directory = "../Data/Symptoms/Total/"
train_directory = "../Data/Symptoms/Train/"
test_directory = "../Data/Symptoms/Test/"

def split_each_file(directory,train_directory,test_directory):
    for fn in os.listdir(directory):
        print("Filename : ",fn)
        data = pd.read_csv(directory+fn)
        print("Number of samples in the dataset :",data.shape[0])
        train,test = split_data(data)
        print("Number of samples in the train data :",train.shape[0])
        print("Number of samples in the test data : ",test.shape[0])
        train.to_csv(train_directory+fn,index = False)
        test.to_csv(test_directory+fn,index = False)

split_each_file(directory,train_directory,test_directory)
