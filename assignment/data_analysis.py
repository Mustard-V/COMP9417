import arff, numpy as np
import pandas as pd
import numpy
from sklearn.base import TransformerMixin
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
import warnings
#this part is used for insert data and Analyze the data, in model part we will create the model
###initial inport data and classification it 
###notes that I use windows os for this project, therefore fileslocatioin are differnet in linux.
###you need to change these two lines to read file.
def insert_data():
    train_set = pd.read_csv('D:\\2020T2\\comp9417\COMP9417\\assignment\\train.csv')
    test_set = pd.read_csv('D:\\2020T2\\comp9417\\COMP9417\\assignment\\test.csv')
    ###make it easier to oprate data.
    train_set.index = train_set['id']
    test_set.index = test_set['id']

    #print("demension of train"+ str(train_set.shape) + "demension of test " + str(test_set.shape))

    ###try linear regression to make the whole algrithem run.
    ###then use knn algrithem train the data remember to use cross validation to redeuce over-fit 
    ###test and print top five list
    #print(train_set.head(5))

    #print(test_set.head(5))
    return train_set, test_set

###this is used for initial linear regression
def create_data_for_linear_regression(df):
    # a.) Use the `id` feature as the index column of the data frame
    df = df.set_index('id')

    # b.) Only use easy to process features
    #  Warning: huge information loss here, you should propably include more features in your production code.
    # production companies
#all_movies["companies_list"] = all_movies["production_companies"].apply(
   # get_list_of_values, args=('name',))
#most_cmn_comps = find_most_common("companies_list", 10)
#one_hot_encode_most_common("production_companies", "companies_list", most_cmn_comps)

#记得使用mostcommon和记得把dummies之后的内容，把原来的内容给删掉要不然影响内容
    df = df[['budget', 'original_language' ,'popularity', 'runtime', 'status','production_companies','production_countries','spoken_languages']]
    
    # c.) One-Hot-Encoding for all nominal data
    df = pd.get_dummies(df)
    print(df.head(10))
    # d.) The `runtime` feature is not filled in 2 of the rows. We replace those empty cells / NaN values with a 0.
    #  Warning: in production code, please use a better method to deal with missing cells like interpolation or additional `is_missing` feature columns.
    return df.fillna(0)

