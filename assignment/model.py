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
###initial inport data and classification it 
train_set = pd.read_csv('D:\\2020T2\\comp9417\COMP9417\\assignment\\train.csv')
test_set = pd.read_csv('D:\\2020T2\\comp9417\\COMP9417\\assignment\\test.csv')
train_set.index = train_set['id']
test_set.index = test_set['id']

print("demension of train"+ str(train_set.shape) + "demension of test " + str(test_set.shape))
###try linear regression to make the whole algrithem run.
###then use knn algrithem train the data remember to use cross validation to redeuce over-fit 
print(train_set.head(5))

print(test_set.head(5))

###use test set to show the result