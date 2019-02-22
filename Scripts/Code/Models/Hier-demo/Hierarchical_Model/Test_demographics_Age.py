#Test the pop-ware model
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,roc_curve
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import random
random.seed(1)
from sklearn.linear_model import Lasso


# * The symptoms included are as follows:

symptoms = ['intercept',
            'fever',
            'sorethroat',
            'cough',
            'muscle',
            'virus']
aucs_ = defaultdict()

print(symptoms)

def read_file(filename):
    data = pd.read_csv(filename)
    data['intercept'] = 1
    columns = list(data.columns)
    columns = columns[-1:] + columns[:-1]
    data = data[columns]
#     train_data = data.drop(['virus'],axis =1).as_matrix()
    return data

def read_parameters(filename):
    parameters = pd.read_csv(filename)
    return parameters


# #### Get the parameters for the different dataset combinations


directory_ = "./Parameters_Age/"
with_demographics_ = ['with_demographics_goviral.csv']
with_demographic_parameters = defaultdict()


def return_parameters(file,parameters_of):
    param = read_parameters(file)
    parameter_dict = defaultdict()
    for i in parameters_of:
        parameter_dict[i] = list(param[i])
    return parameter_dict
    

def get_parameters(dataset_name,parameters):
    return np.array(list(parameters[dataset_name]))


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def get_results(param,sample_points):
    return sigmoid(np.dot(param,sample_points.T)  )


def save_results_for_finding_threshold(filename,dataframe,predicted):
    results = pd.DataFrame()
    results['Actual'] = dataframe['virus']
    results['Predicted'] = predicted
    print(results.head())
    results.to_csv(filename,index = False)


def get_all_datasets(training_data_,training_directory):
    datasets = defaultdict()
    for i in training_data_:
        data = read_file(training_directory+i)
        datasets[i[:-4]] = (data)
    return datasets


def get_all_results(data_dict,param):
    results = defaultdict()
    for i in list(param.keys()):
        data,train = data_dict[i]
        results[i] = get_results(param[i],train)
    return results



def result_statistics(list_):
#     print("Min : ",min(list_))
#     print("Max : ",max(list_))
#     print("Mean : ",np.mean(list_))
#     print("Standard Deviation : ",np.std(list_))
    return min(list_),max(list_)


def return_class(threshold,list_):
    ans = [1 if x >= threshold else 0 for x in list_]
    return ans

def metrics_pred(list1,list2):
    # f1 =f1_score(list1,list2)
    # precision = precision_score(list1,list2)
    # recall = recall_score(list1,list2)
    # accuracy = accuracy_score(list1,list2)
    fpr,tpr,threshold = roc_curve(list1,list2)
    auc = metrics.auc(fpr,tpr)
#     print("f1 score : ",f1)
#     print("Precision score : ",precision)
#     print("Recall : ",recall)
#     print("Accuracy : ",accuracy)
#     print("Area under the curve : ",auc)
    return auc


def find_threshold(min_,max_,list1,list2,step_size = 1e-3):
    auc_thresholds = defaultdict()
    value = min_
    while value < max_:
        auc_thresholds[value] = metrics_pred(list1,return_class(value,list2))
        value += step_size
    optimal_threshold = max(auc_thresholds.items(), key=lambda x: x[1]) 
    return optimal_threshold


def get_threshold(pred,true):
    min_,max_ = result_statistics(pred)
    threshold = find_threshold(min_,max_,true,pred)
    return threshold


def return_all_thresholds(results,data,y_true):
    thresholds = defaultdict()
    for i in list(data.keys()):
        print("_____________________")
        min_,max_ = result_statistics(results[i])
        
        threshold = find_threshold(min_,max_,y_true[i],results[i])
        print("Found threshold for : ",i)
        thresholds[i] = threshold
    return thresholds


def test(filename_,param,thresholds_):
    aucs = defaultdict()
    data,train = read_file(filename_)
    for i in list(param.keys()):
        test_results = get_results(param[i],train)
        auc_ = metrics_pred(data['virus'],return_class(thresholds_[i][0],test_results))
        aucs[i] = auc_
    return aucs


def return_final_auc_scores(training_data_,training_directory,filename_,parameters):
    data = get_all_datasets(training_data_)
    results = get_all_results(data,parameters)
    #find the thresholds
    thresholds = return_all_thresholds(results,data)
    #get the auc values
    aucs_= test(filename_,parameters,thresholds)
    return aucs_

def create_dict(dict_):
    temp = []
    for k,v in dict_.items():
        temp.append((k,v))
    return temp
        


results_symp = defaultdict()
results_demo = defaultdict()


# #### Get the symptoms

def get_gender(dataframe_):
    df = dataframe_[['male','female']]
    temp = df.apply(lambda x:x.argmax(),axis =1)
    return temp


def get_age(dataframe_):
    df = dataframe_[['age 0-4', 'age 5-15', 'age 16-44', 'age 45-64', 'age 65+']]
    temp = df.apply(lambda x: x.argmax(), axis=1)
    return temp


def get_predictions_all(name,train,only_symp_age,only_symp_gender,param_dict,temp_age,temp_gender,collection_mode = 'clinically_collected',population ='population'):
    results = []
    for i in range(train.shape[0]):
        result = []
        sample_point = train[i,:]
        gender = list(only_symp_gender.iloc[i][:])
        age = list(only_symp_age.iloc[i][:])
        p_data = get_results(param_dict[name],sample_point)
        result.append(p_data)
#         result.append(gender[0]*get_results(param_dict['male'],sample_point))
#         result.append(gender[1]*get_results(param_dict['female'],sample_point))

        result.append(age[0]*get_results(param_dict['age 0-4'],sample_point))
        result.append(age[1]*get_results(param_dict['age 5-15'],sample_point))
        result.append(age[2]*get_results(param_dict['age 16-44'],sample_point))
        result.append(age[3]*get_results(param_dict['age 45-64'],sample_point))
        result.append(age[4]*get_results(param_dict['age 65+'],sample_point))
        
#         p_collection = get_results(param_dict[collection_mode],sample_point)
#         p_gender = get_results(param_dict[temp_gender[i]],sample_point)
#         p_age = get_results(param_dict[temp_age[i]],sample_point)
# #         p_population = get_results(param_dict[population],sample_point)
#         result = [p_data,p_collection,p_gender+p_age]
        results.append(result)
    return results
    
    


def get_coeff(X,Y):
    lm = linear_model.LogisticRegression()
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 10)
    lm.fit(x_train,y_train)
    y_pred = lm.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
#     print("Accuracy :",acc)
    fpr,tpr,threshold = roc_curve(y_test,y_pred)
    auc_score = metrics.auc(fpr,tpr)
#     print("AUC :",auc_score)
    coefficients = lm.coef_.tolist()[0]
    print("Coefficients : ",coefficients)
    intercept = lm.intercept_.tolist()[0]
    return coefficients,intercept
    


def norm(list_):
    min_ = min(list_)
    max_ = max(list_)
    denom = max_ - min_
    ans = [x-min_/denom for x in list_]
    return ans



COLLECTION_MODE = {'nyumc':'clinically_collected',
                   'goviral':'individually_reported',
                   'fluwatch':'individually_reported',
                   'hongkong': 'health_worker',
                   'hutterite':'health_worker','loeb':'health_worker'}


def process_all(training_data_list,training_directory,filename_,parameters,collection_mode = COLLECTION_MODE):
    name_dataset = filename_.split('/')[-1]
    thresholds = defaultdict()
    print(name_dataset)
    data = get_all_datasets(training_data_list,training_directory)
    print("Got the data")
    print("Now finding coefficients for the the datasets!")
    weights = defaultdict()
    for i in data.keys():
        print("Analyzing the dataset : ",i)
        data_ = data[i]
        temp_age = get_age(data_)
        temp_gender = get_gender(data_)
        only_symp_data = data_[symptoms]
        only_symp_gender = data_[['male','female']]
        only_symp_age = data_[['age 0-4','age 5-15','age 16-44','age 45-64','age 65+']]
        only_symp_data.drop('virus',axis = 1,inplace = True)
        train_data_symp = only_symp_data.as_matrix()
        prediction = get_predictions_all(i,train_data_symp,only_symp_age,only_symp_gender,parameters,temp_age,temp_gender,COLLECTION_MODE[i])
        y_true = list(data_['virus'])
        coefficient,intercept = get_coeff(prediction,y_true)
        weights[i] = (coefficient,intercept)
        value = np.array(np.dot(prediction,np.array(weights[i][0]).T)+weights[i][1])
        values = [sigmoid(j) for j in value]
        
        
        threshold = get_threshold(values,y_true)
        print("Found threshold for ",i)
        thresholds[i] = threshold[0]
        ans = [(y_true[i],values[i]) for i in range(len(y_true))]
    return weights,thresholds,ans


# In[27]:


def process_test(training_directory,filename_,parameters,weights,thresholds,collection_mode = COLLECTION_MODE):
    aucs_ = defaultdict()
    predictions = defaultdict()
    test_data = get_all_datasets([filename_],training_directory)
    name = filename_.split('.')[0]
    print("Name : ",name) 
    data_ = test_data[name]
    temp_age = get_age(data_)
    temp_gender = get_gender(data_)
    only_symp_data = data_[symptoms]
    only_symp_data.drop('virus',axis = 1,inplace = True)
    only_symp_gender = data_[['male','female']]
    only_symp_age = data_[['age 0-4','age 5-15','age 16-44','age 45-64','age 65+']]
    y_true = list(data_['virus'])
    train_data_symp = only_symp_data.as_matrix()
    for i in weights.keys():
        print("Using the parameters of : ",i)
        prediction = get_predictions_all(i,train_data_symp,only_symp_age,only_symp_gender,parameters,temp_age,temp_gender,COLLECTION_MODE[i])
#         temp = [i[1:] for i in prediction]
#         first = [i[0] for i in prediction]
        prediction = np.array(prediction)
        value = np.array(np.dot(prediction,np.array(weights[i][0]).T)+weights[i][1])
        values = [sigmoid(i) for i in value]
        predictions[i] = values
   
    print("Got the predicitions from the different parameters")
    for i in weights.keys():
        auc_ = metrics_pred(y_true,return_class(thresholds[i],predictions[i]))
        aucs_[i] = auc_
        print("Found the auc for ",i)
    return aucs_
#         


# #### With demographics

# ##### Generating the results for NYUMC

# In[28]:


training_data_nyumc = ['goviral.csv']
training_directory = "../../Data/Symptoms_Demo/Goviral/Train/"
testing_directory = "../../Data/Symptoms_Demo/Goviral/Test/"
filename_ = 'goviral.csv'


# In[29]:


cols = ['goviral', 'age 0-4','age 5-15','age 16-44','age 45-64','age 65+', 'population']
demo_nyumc = return_parameters(directory_+'with_demographics_goviral.csv',cols)
demo_nyumc.keys()


# In[30]:


print("With demographics!")


# In[31]:


weights_nyumc,thresholds_nyumc,pred = process_all(training_data_nyumc,training_directory,filename_,demo_nyumc)


# In[32]:


thresholds_nyumc


# In[33]:


weights_nyumc


# In[34]:


aucs_nyumc = process_test(testing_directory,filename_,demo_nyumc,weights_nyumc,thresholds_nyumc)


# In[ ]:





# In[35]:


aucs_nyumc


# In[141]:


aucs_['goviral'] = aucs_nyumc


# ##### Generating the results for FluWatch

# In[36]:


training_data_goviral = ['fluwatch.csv']
training_directory = "../../Data/Symptoms_Demo/Fluwatch/Train/"
testing_directory = "../../Data/Symptoms_Demo/Fluwatch/Test/"
# training_directory = "../../Data/Symptoms_Demo/Balanced_Data/Train/"
filename_ = 'fluwatch.csv'


# In[40]:


cols = ['fluwatch', 'age 0-4','age 5-15','age 16-44','age 45-64','age 65+', 'population']
demo_goviral = return_parameters(directory_+'with_demographics_fluwatch.csv',cols)
demo_goviral.keys()


# In[41]:


weights_goviral,thresholds_goviral,ans = process_all(training_data_goviral,training_directory,filename_,demo_goviral)


# In[42]:


aucs_goviral1 = process_test(testing_directory,filename_,demo_goviral,weights_goviral,thresholds_goviral)


# In[43]:


aucs_goviral1


# In[44]:


aucs_['nyumc'] = aucs_goviral1


# 
# ##### Generating the results for hongkong

# In[50]:


training_data_fluwatch = ['hongkong.csv']
training_directory = "../../Data/Symptoms_Demo/Hongkong/Train/"
testing_directory = "../../Data/Symptoms_Demo/Hongkong/Test/"
# training_directory = "../../Data/With_Improved_Target/With_Demographics/"
filename_ = 'hongkong.csv'


# In[52]:


cols = ['hongkong', 'age 0-4','age 5-15','age 16-44','age 45-64','age 65+', 'population']
demo_fluwatch = return_parameters(directory_+'with_demographics_hongkong.csv',cols)
demo_fluwatch.keys()


# In[53]:


weights_fluwatch,thresholds_fluwatch,ans = process_all(training_data_fluwatch,training_directory,filename_,demo_fluwatch)


# In[54]:


aucs_fluwatch1 = process_test(testing_directory,filename_,demo_fluwatch,weights_fluwatch,thresholds_fluwatch)


# In[55]:


aucs_fluwatch1


# In[56]:


aucs_['hongkong'] = aucs_fluwatch1


# ##### Generating the results for Hutterite

# In[57]:


training_data_hongkong = ['hutterite.csv']
# training_directory = "../../Data/With_Improved_Target/With_Demographics/"
training_directory = "../../Data/Symptoms_Demo/Hutterite/Train/"
testing_directory = "../../Data/Symptoms_Demo/Hutterite/Test/"
filename_ = 'hutterite.csv'


# In[58]:


cols = ['hutterite', 'age 0-4','age 5-15','age 16-44','age 45-64','age 65+', 'population']
demo_hongkong = return_parameters(directory_+'with_demographics_hutterite.csv',cols)
demo_hongkong.keys()


# In[59]:


weights_hongkong,thresholds_hongkong,ans = process_all(training_data_hongkong,training_directory,filename_,demo_hongkong)


# In[60]:


aucs_hongkong = process_test(testing_directory,filename_,demo_hongkong,weights_hongkong,thresholds_hongkong)


# In[61]:


aucs_hongkong


# In[62]:


aucs_['hutterite'] = aucs_hongkong


# ### Test Loeb

# In[64]:


training_data_loeb = ['loeb.csv']
# training_directory = "../../Data/With_Improved_Target/With_Demographics/"
training_directory = "../../Data/Symptoms_Demo/Loeb/Train/"
testing_directory = "../../Data/Symptoms_Demo/Loeb/Test/"
filename_ = 'loeb.csv'


# In[69]:


cols = ['loeb', 'age 0-4','age 5-15','age 16-44','age 45-64','age 65+', 'population']
demo_loeb = return_parameters(directory_+'with_demographics_loeb.csv',cols)
demo_loeb.keys()


# In[70]:


weights_loeb,thresholds_loeb,ans = process_all(training_data_loeb,training_directory,filename_,demo_loeb)


# In[71]:


aucs_loeb = process_test(testing_directory,filename_,demo_loeb,weights_loeb,thresholds_loeb)


# In[72]:


aucs_loeb


# In[73]:


aucs_['loeb'] = aucs_loeb


# In[ ]:





# In[74]:


aucs_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




