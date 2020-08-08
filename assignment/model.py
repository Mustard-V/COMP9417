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
import data_analysis


train_data, test_data = data_analysis.insert_data();

# 1.) Extract the target variable `revenue` and use the `id` column as index of that data frame
df_trainval_y = train_data[['id','revenue']].set_index('id')

# 2.) Prepare the training and test data by using the function we defined above
df_trainval_X = data_analysis.create_data_for_linear_regression(train_data)
df_test_X  = data_analysis.create_data_for_linear_regression(test_data)

# 3.) Create columns in train/test dataframes if they only exist in one of them (can happen through one hot encoding / get_dummies)
#  Example: There are no status=`Post Production` entries in the training set, but there are some in the test set.
df_trainval_X, df_test_X = df_trainval_X.align(df_test_X, join='outer', axis=1, fill_value=0)

# 4.) Show the first rows of one of the prepared tables
df_trainval_X.head(2)
