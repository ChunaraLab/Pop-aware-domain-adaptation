import pandas as pd

def sampling(dataset):
    #sampling logic goes here for sampling the different datasets
    return True


def read_file(filename):
    #function to read the file
    data = pd.read_csv(filename)
    data['intercept'] = 1
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    data.drop(['male','female','age 0-4','age 5-15','age 16-44','age 45-64','age 65+'],axis=1,inplace = True)
    print(data.columns)
    return data

def split_data(dataset):
    #function to split the data
    #not using sklearn train test split as column names are not returned
    train = dataset.sample(frac = 0.8, random_state = 100)
    test = dataset.drop(train.index)
    test.to_csv('./Test/'+filename_to_store_test)
    return train,test

def get_data(directory,datasets_,files_):
    for i in files_:
        filename = directory+i
        data = read_file(filename)
        #sampling here
        # train_data = split_data(data,directory_to_store+i) data is already split now
        temp = i[:-4]
        datasets_[temp] = data
    print("All data read!")
    return datasets_
