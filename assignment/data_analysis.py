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
    #print(test_set.head(5))
    return train_set, test_set

###this is used for initial linear regression
def create_data_for_linear_regression(train_set, test_set):
    
    train_set = find_feature(train_set)
    test_set = find_feature(test_set)
    #after we decide the veriable to use, we need to dummy these value make it easier fo knn 
    train_set = pd.get_dummies(train_set)
    test_set = pd.get_dummies(test_set)

    print(test_set.columns)
    
    return train_set.fillna(0), test_set.fillna(0)

def find_feature(train_set):
    #there are 22 variable column in dataset, we need to We need to figure out what is necessary， otherwise the dataset is too complex.
   
    #transfer relase date to dummy months...form Box Office Beginner
    train_set.loc[train_set['release_date'].isnull(), 'release_date'] = '0/0/00'
    train_set["release_month"] = train_set["release_date"].apply(lambda x: x.split("/")[0])
    #print(train_set["release_month"].value_counts())
    dummy_months = pd.get_dummies(train_set["release_month"], prefix="month")

    #month_pivot = train_set.pivot_table(index="release_month", values="revenue", aggfunc=np.mean)
    train_set = pd.concat([train_set[['budget', 'original_language' ,'popularity', 'runtime', 'status','production_companies','production_countries','spoken_languages']], dummy_months], axis=1)

    #replace 
    return train_set

#train_set, test_set = insert_data()
#create_data_for_linear_regression(train_set, test_set)
