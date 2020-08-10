import arff, numpy as np
import pandas as pd
import numpy
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
import warnings
import data_analysis

class knn_model:

    #get data from csv
    train_data, test_data = data_analysis.insert_data()
    train_y = train_data[['id','revenue']]
    train_y = train_y.set_index("id")

    # get filterd,analysised dummied data
    train_X, test_X = data_analysis.create_data_for_linear_regression(train_data,test_data)

    #Eliminate the differences between the two tables
    train_X, test_X = train_X.align(test_X, join='outer', axis=1, fill_value=0)

    train_X_value = train_X.values
    train_Y_value = train_y.values
    test_X_value  = test_X.values
    
    #Validation
    X_train, X_val, y_train, y_val = train_test_split(train_X_value, train_Y_value, test_size=0.7, random_state=56)

    #Scale_X, thanks alexander's great work in https://www.kaggle.com/alexandermelde/code-template-for-simple-regression-prediction
    #really useful, We need the scale input value and the y value to reduce the dependency, Improve the accuracy of the model
    X_scaler = StandardScaler()
    train_X_scaled  = X_scaler.fit_transform(X_train)
    val_X_scaled    = X_scaler.transform(X_val)
    X_test_scaled   = X_scaler.transform(test_X_value)
    ###scale y
    y_scaler = MinMaxScaler((0,1))
    y_train_scaled  = y_scaler.fit_transform(np.log(y_train)).ravel() 
    y_val_scaled  = y_scaler.transform(np.log(y_val)).ravel() 
    #print( y_train_scaled)

    # get the model
    reg = KNeighborsRegressor(n_neighbors=3).fit(train_X_scaled, y_train_scaled)
    
    #apply the model on val
    val_result = reg.predict(val_X_scaled)
    j = 0
    i = 0
    for k1 in val_result:
        if(  y_val_scaled[j]*0.7 < k1 <  y_val_scaled[j] *1.3):
            i = i + 1
            j = j + 1
        else:
            j = j + 1
    print("The accuracy in validation set " , i/j)
    #####looks like ok


    #apply the model on test
    test_result = reg.predict(X_test_scaled)
    #reverse the scaled y in to revenue
    y_test_pred = np.exp(y_scaler.inverse_transform(np.reshape(test_result, (-1,1))))
    print("check point" )
    #print(test_y_scaled[0])
############################################
###### Write the result to submission.csv
    # 1.) Add the predicted values to the original test data
    test_result = test_data.assign(revenue=y_test_pred)

# 2.) Extract a table of ids and their revenue predictions
    df_test_y = test_result[['id','revenue']].set_index('id')

# 3.) save that table to a csv file. On Kaggle, the file will be visible in the "output" tab if the kernel has been commited at least once.
    df_test_y.to_csv("submission.csv")


#run the model
k1 = knn_model