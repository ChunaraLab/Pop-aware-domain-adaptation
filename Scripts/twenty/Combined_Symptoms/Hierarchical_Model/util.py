import pandas
from scipy.misc import logsumexp
import numpy as np
# from data_preprocessing import *

TARGET_VARIABLE = 'virus'
# FEATURES = get_features()


def get_positive_datapoints(dataset):
    #get the datapoints from the dataset where flu is positive
    return dataset.loc[dataset[TARGET_VARIABLE] == 1]

def measure(data_points,j,sample_points):
    #measure of some particular symptom in the dataset where it contributes to flu
    data = data_points.loc[data_points[j] == 1]
    value = round(data.shape[0]/sample_points,7)
    return value

def get_fraction_positive(dataset):
    #get the fraction of flu positives in the dataset
    original_dataset = dataset.shape[0]
    new_dataset = get_positive_datapoints(dataset).shape[0]
    return int(new_dataset/original_dataset)

def get_denominator_multinomial(parameters):
    #get the denominator of the multinomial
    denomiator = logsumexp(np.asarray(parameters))
    return denomiator

def multinomial_parameter_difference(data_parameter,denominator):
    #multinomial parameter difference
    return data_parameter - denominator

def get_divergence(parameters,parent,divergence_function):
    #calulcate the l1 divergence between the child and parent parameters
    if divergence_function == 1:
        l1norm = np.linalg.norm((np.asarray(parameters) - np.asarray(parent)),ord = 1)
        return l1norm

    #calculate the l2 divergence between the child and the parent parameters
    elif divergence_function == 2:
        l2norm = np.linalg.norm((np.asarray(parameters) - np.asarray(parent)),ord = 2)
        return l2norm


def prepare_data(datasets_,positive_data):
    #returns the positive examples from all the dataset_specific
    #we could also include some  sampling logic here to enhance the data selection process
    for key,value in datasets_.items():
        key = key[:-4]
        positive_data[key] = get_positive_datapoints(value)
    return positive_data
