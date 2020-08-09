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

class linear_model:
    train_data, test_data = data_analysis.insert_data();

    # 1.) Extract the target variable `revenue` and use the `id` column as index of that data frame
    df_trainval_y = train_data[['id','revenue']]
    df_trainval_y = df_trainval_y.set_index("id")
    # 2.) Prepare the training and test data by using the function we defined above
    df_trainval_X = data_analysis.create_data_for_linear_regression(train_data)
    df_test_X  = data_analysis.create_data_for_linear_regression(test_data)

    # 3.) Create columns in train/test dataframes if they only exist in one of them (can happen through one hot encoding / get_dummies)
    #  Example: There are no status=`Post Production` entries in the training set, but there are some in the test set.
    df_trainval_X, df_test_X = df_trainval_X.align(df_test_X, join='outer', axis=1, fill_value=0)

    # 4.) Show the first rows of one of the prepared tables
    df_trainval_X.head(2)
    # 1.) Remove table meta data, column names etc. → Just use values for prediction.
    X_trainval = df_trainval_X.values
    y_trainval = df_trainval_y.values

    X_test  = df_test_X.values

    # 2.) Create Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, random_state=56)

    # 3.) Scale
    X_scaler = StandardScaler()
    X_train_scaled  = X_scaler.fit_transform(X_train)
    X_val_scaled    = X_scaler.transform(X_val)

    X_test_scaled   = X_scaler.transform(X_test)

    y_scaler = MinMaxScaler((0,1)) # transform and convert column-vector y to a 1d array with ravel
    y_train_scaled  = y_scaler.fit_transform(np.log(y_train)).ravel() 
    #y_val_scaled  = y_scaler.transform(np.log(y_val)).ravel() #not used but here for consistency

    # 4.) Calculate the coefficients of the linear regression / "Train"
    reg = KNeighborsRegressor().fit(X_train_scaled, y_train_scaled)

    # 5.) Define functions to calculate a score
    def score_function(self, y_true, y_pred):
        # see https://www.kaggle.com/c/tmdb-box-office-prediction/overview/evaluation
        # we use Root Mean squared logarithmic error (RMSLE) regression loss
        assert len(y_true) == len(y_pred)
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

    def score_function2(self,y_true, y_pred):
        # alternative implementation
        y_pred = np.where(y_pred>0, y_pred, 0)
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    def inverseY(self,y):
        return np.exp(self.y_scaler.inverse_transform(np.reshape(y, (-1,1))))

# 6.) Apply the regression model on the prepared train, validation and test set and invert the logarithmic scaling
    k = reg.predict(X_train_scaled)
    print(k)
    print(np.exp(y_scaler.inverse_transform(np.reshape(k, (-1,1)))))
    #y_train_pred  = inverseY(reg.predict(X_train_scaled))
    #y_val_pred    = inverseY(_,reg.predict(X_val_scaled))
    #y_test_pred   = inverseY(reg.predict(X_test_scaled))
                   
# 7.) Print the RMLS error on training, validation and test set. it should be as low as possible
    #print("RMLS Error on Training Dataset:\t", score_function(y_train , y_train_pred), score_function2(y_train, y_train_pred))
    #print("RMLS Error on Val Dataset:\t", score_function(y_val , y_val_pred), score_function2(y_val , y_val_pred))
    #print("RMLS Error on Test Dataset:\t Check by submitting on kaggle")
