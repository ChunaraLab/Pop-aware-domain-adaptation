import numpy as np
import scipy
from scipy import optimize
from util import *

#defining the objective function here
#includes the data dependent objective funtion as well as the divergence/dissimilarity function
#the data dependent function encourages the parameters learnt from the data to correctly represent the features of the data
#the divergence function determines the data flow between the different levels of the hierarchy

def data_dependent(data_points,data_parameters,symptoms,alpha):
    #define the data dependent objective to learn the data_parameters
    data_objective = 0
    #number of samples in the dataset
    sample_points = data_points.shape[0]
    #get the dataset where the flu is positive
    data_points = get_positive_datapoints(data_points)
    # for i in data_points:
    denomiator_for_multinomial = get_denominator_multinomial(data_parameters)
    for index,j in enumerate(symptoms):
        #measure  : measure of the symptom contributing to flu in the particular dataset
        #alpha is the regularizing parameter
        #if alpha is 1 Laplace smoothing
        #parameter specific : capture the relation between the different symptoms
        #multinomial_parameter_difference uses the multinomial distribution
        dataset_specific = measure(data_points,j,sample_points) + alpha
        parameter_specific = multinomial_parameter_difference(data_parameters[index],denomiator_for_multinomial)
        data_objective += (2*dataset_specific+parameter_specific)
    return data_objective


def divergence(parameters,parent,divergence_function):
    #define the divergence between the parameters and the parent parameters
    divergence = get_divergence(parameters,parent,divergence_function)
    #logic for the divergence
    return divergence

def divergence_all_pairs(levels,influence,parents,parameters,divergence_function):
    divergence_value = 0
    for level in levels:
        # print("Level : ",level)
        for node in levels[level]:
            # print("Node : ",node)
            influence_for_node = influence[node]
            for par in parents[node]:
                # print("Parent : ",par)
                # influence_index = parents[node].index(par)
                divergence_value += influence_for_node[0] * divergence(parameters[node],parameters[par],divergence_function)
    return divergence_value

def data_dependent_all_datasets(datasets,parameters,symptoms,alpha):
    all_datasets_value = 0
    for i,value in datasets.items():
        all_datasets_value += data_dependent(datasets[i],parameters[i],symptoms,alpha)
    return all_datasets_value

def joint_objective(datasets,parameters,symptoms,alpha,levels,influence,parents,divergence_function):
    joint_objective = -(data_dependent_all_datasets(datasets,parameters,symptoms,alpha)) + divergence_all_pairs(levels,influence,parents,parameters,divergence_function)
    print(joint_objective)
    return joint_objective
