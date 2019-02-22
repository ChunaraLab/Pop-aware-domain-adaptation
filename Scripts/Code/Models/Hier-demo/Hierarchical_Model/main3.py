import pandas as pd
import numpy as np
import os
import time
import ast
import configparser
import random

from collections import defaultdict
from util import *
from objective_function import *
from scipy.optimize import minimize
from data_preprocessing import *
from scipy import optimize

random.seed(1)
datasets_ = defaultdict() #place holder to store the different datasets to be used for training
positive_data  = defaultdict()
data_parameters = defaultdict()
levels = defaultdict()
parents = defaultdict()
parent_parameters = defaultdict()
influence = defaultdict()

#configparser to read the configurations for the experiment
def get_dataset_parameters(config,name_,files_):
    for i in files_:
        i = i[:-4]
        data_parameters[i] = list(map(float,ast.literal_eval(config.get(name_,i))))
    return data_parameters

def get_levels(config,name_,levels_):
    for i in levels_:
        levels[i] = ast.literal_eval(config.get(name_,i))
    return levels

def get_parents(config,name_,parents_):
    for i in parents_:
        parents[i] = ast.literal_eval(config.get(name_,i))
    return parents

def get_parent_parameters(config,name_,parent_parameters_):
    for i in parent_parameters_:
        parent_parameters[i] =  list(map(float,ast.literal_eval(config.get(name_,i))))
    return parent_parameters

def get_influence(config,name_,parents_):
    for i in parents_:
        influence[i] = list(map(float,ast.literal_eval(config.get(name_,i))))
    return influence



def get_all_configurations():
    #get the configurations from the configuration file
    config = configparser.ConfigParser()
    config.read("./Gender_Hyperparameters/with_demographics_hutterite.ini")
    print("Read the configuration file!")

    method = ast.literal_eval(config.get("general","method"))

    directory_dataset_name = ast.literal_eval(config.get("general","directory_datasets"))

    directory_store_params = ast.literal_eval(config.get("general","store_parameters"))

    parameter_names = ast.literal_eval(config.get("data","parameter_names"))

    divergence_function = ast.literal_eval(config.get("general","DIVERGENCE_FUNCTION"))
    # print("Divergence function : ",divergence_function)

    files_ = ast.literal_eval(config.get("data","files_to_read"))
    # print("Files to be read are : ",files_)

    levels_ = ast.literal_eval(config.get("data","levels"))
    # print("Levels are : ",levels_)

    parents_ = ast.literal_eval(config.get("data","parents"))
    # print("Parents are : ",parents_)

    parent_parameters_ = ast.literal_eval(config.get("data","parent_parameters"))
    # print("Parent Parameters are : ",parent_parameters_)

    symptoms = ast.literal_eval(config.get("data","SYMPTOMS"))
    print("Symptoms are : ",symptoms)

    alpha = ast.literal_eval(config.get("data","ALPHA"))
    # print("Alpha is : ",alpha)

    data_parameters = get_dataset_parameters(config,"dataset_parameters",files_)
    # print("Data parameters are : ",data_parameters)

    levels = get_levels(config,"levels",levels_)
    # print("The levels are : ",levels)

    parents = get_parents(config,"parents",parents_)
    # print("Parents are : ",parents)

    parent_parameters = get_parent_parameters(config,"parent_parameters",parent_parameters_)
    # print("Parent parameters are : ",parent_parameters)

    influence = get_influence(config,"influence",parents_)

    # print("Influence is : ",influence)

    return method,directory_dataset_name,directory_store_params,parameter_names,divergence_function,files_,levels_,parents_,parent_parameters_,symptoms,alpha,data_parameters,levels,parents,parent_parameters,influence

#get all the configuration parameters
method,directory_dataset_name,directory_store_params,parameter_names,divergence_function,files_,levels_,parents_,parent_parameters_,symptoms,alpha,data_parameters,levels,parents,parent_parameters,influence = get_all_configurations()
print("Got all the configurations!")
#file directory
directory = directory_dataset_name
#directory to store the files
directory_to_store = "../Data/Test/"

#get all the datasets
#this is a dictionary : dataset name as the key and the dataframe as the value
datasets_ = get_data(directory,datasets_,files_)
#positive data is similar to datasets_ but only includes the positive data points in each dataset
positive_data = prepare_data(datasets_,positive_data)


def get_parameters(symptoms,parameter_names,x):
    #get the parameters from the flattened list
    #the flattened list is needed for minimizing the objective function
    no_symptoms = len(symptoms)
    param = defaultdict()
    start = 0

    for i in parameter_names:
        param[i] = x[start:start+no_symptoms]
        start += no_symptoms
    return param

def generate_params_for_minimizing(data_parameters,parent_parameters,parameter_names,x):
    #create a flattened list from all the parameters
    parameters = dict(list(data_parameters.items()) + list(parent_parameters.items()))
    for i in parameter_names:
        x.append(parameters[i])
    temp = [item for sublist in x for item in sublist]
    return temp

x = []
x = generate_params_for_minimizing(data_parameters,parent_parameters,parameter_names,x)

args = (parameter_names,datasets_,symptoms,alpha,levels,influence,parents,divergence_function)

def function_to_minimize(x,*args):
    #get the arguments
    parameter_names,datasets_,symptoms,alpha,levels,influence,parents,divergence_function = args
    #get the parameters
    parameters = get_parameters(symptoms,parameter_names,x)
    return joint_objective(datasets_,parameters,symptoms,alpha,levels,influence,parents,divergence_function)

# parameters = dict(list(data_parameters.items()) + list(parent_parameters.items()))
# x = joint_objective(datasets_,parameters,symptoms,alpha,levels,influence,parents,divergence_function)

result = minimize(function_to_minimize,x0 = np.asarray(x),method=method,tol = 1e-6,args = args,options={'maxfev':10**5, 'disp': True})
print(result)

parameters_final = get_parameters(symptoms,parameter_names,result.x)
print(parameters_final)


def store_parameters(parameters_final,parameter_names,directory):
    store_parameters = pd.DataFrame()
    store_parameters['symptoms'] = list(symptoms)
    for i in parameter_names:
        store_parameters[i] = list(parameters_final[i])
    store_parameters.to_csv(directory,index = False)

store_parameters(parameters_final,parameter_names,directory_store_params)
