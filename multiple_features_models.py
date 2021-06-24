import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#.................Result Folder making.......................
if not os.path.exists('Multiple_features_model_results'):
    os.makedirs('Multiple_features_model_results')
#...........Reading the data.................................................

data = pd.read_excel('Bases_Final_ADS_Jun2021.xlsx')     #reading the dataset

ind_se_dt = pd.DataFrame(data[['data_tidy', 'ind_se','renda_r', 'pop_ocup_br', 'massa_r', 'du','pmc_a_se',
        'temp_max_se', 'temp_min_se', 'pmc_r_se', 'pim_se']])

independent_var = ['renda_r', 'pop_ocup_br', 'massa_r', 'du','pmc_a_se',        #Features
        'temp_max_se', 'temp_min_se', 'pmc_r_se', 'pim_se']


target_f = ['ind_se']                   #Target feature

mask = (ind_se_dt['data_tidy'] >= '2004-01-01') & (ind_se_dt['data_tidy'] <= '2021-02-01')
mask2 = (ind_se_dt['data_tidy'] >= '2021-03-01')

dt =  ind_se_dt.loc[mask].reset_index()         #Training data
test_dt = ind_se_dt.loc[mask2].reset_index()    #Future predict_linearion data

#................ Fiilin NULL values with mean value of respective columns.............
renda_mean, massa_mean = dt['renda_r'].mean(), dt['massa_r'].mean()
dt['renda_r'].fillna(value=renda_mean, inplace=True)
dt['massa_r'].fillna(value=massa_mean, inplace=True)

#............. set date as index............................................
dt.set_index('data_tidy', inplace=True) #set date as index
test_dt.set_index('data_tidy', inplace=True) 
test_X = np.array(test_dt[independent_var])
#...........................................................................

dt_X = np.array(dt[independent_var])
dt_y = np.array(dt[target_f])


tscv = TimeSeriesSplit(5)  # 5 fold time series cross validation or train test split

###........................Models.....................................
regr_linear = linear_model.LinearRegression()
regr_svr = SVR(kernel = 'rbf')
regr_rf = RandomForestRegressor(n_estimators=200, max_depth=2, random_state=0)
regr_dt = DecisionTreeRegressor(max_depth=2)
regr_br = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
models_name = ['Linear Regression', 'Support Vector Rrgression', 'Random Fores', 'Decision Tree', 'Bayesian Regression']
models = [regr_linear, regr_svr, regr_rf, regr_dt, regr_br ]

table =pd.DataFrame()
for i, mod in enumerate(models):
    RMSE = []           #root means suqare error
    MSE = []            #mean square error
    R2 = []             #R2 score

    for train_index, valid_index in tscv.split(dt_X):           
        X_train, X_valid = dt_X[train_index], dt_X[valid_index]
        tar_train, tar_valid = dt_y[train_index], dt_y[valid_index]
        mod.fit(X_train, tar_train)                             #model fitting
        predict = mod.predict(X_valid)                           #model prediction
                                       
        RMSE.append(np.sqrt(sum((predict - tar_valid)**2)/len(tar_valid)))
        MSE.append(sum((predict - tar_valid)**2)/len(tar_valid))
        R2.append(r2_score(tar_valid , predict ))
    rmse, mse, r2= np.mean(RMSE), np.mean(MSE), np.mean(R2)

    table =table.append(pd.DataFrame([models_name[i], rmse, mse, r2 ]).T)
table.columns = ["Model name","RMSE","MSE","R2"]
print(table)

table.to_csv('Multiple_features_model_results/Error_evaluation_each model.csv', index = False)

#.............. fitting selected model (SVR)  on all training set for prediction future values.........
regr_svr.fit(dt_X, dt_y)
pred_test_X = regr_svr.predict(test_X)
pred_test_X = pd.DataFrame(pred_test_X)
pred_test_X['data_tidy'] = test_dt.index

pred_test_X.set_index('data_tidy', inplace=True) 
pred_test_X.columns= ['pred_ind_se']

#................ Plotting Predicted Future values ...............................
plt.figure(figsize=(10,5))
plt.plot(pred_test_X, color= 'red', label = 'prediction')
plt.xlabel("Months", fontsize = 12)
plt.ylabel("Consumption (Gwh)", fontsize = 12)
plt.title("Predicted Industrial Energy Consumption Southeast", fontsize = 15)
plt.legend(loc= 'best')
plt.xticks( fontsize = 12)
plt.yticks( fontsize = 12)

plt.savefig('Multiple_features_model_results/predicted_Industrial_Energy_Consumption_Southeast' + '.png', format="PNG" )
plt.show()
#..........plotting predicted values with history data.........................
plt.figure(figsize=(10,5))
plt.plot(dt['ind_se'], color= 'blue', label = 'history')
plt.plot(pred_test_X, color= 'red', label = 'prediction')
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Consumption (Gwh)", fontsize = 12)
plt.title("Predicted Industrial Energy Consumption Southeast", fontsize = 15)
plt.legend(loc= 'best')
plt.xticks( fontsize = 12)
plt.yticks( fontsize = 12)

plt.savefig('Multiple_features_model_results/predicted_history_Industrial_Energy_Consumption_Southeast' + '.png', format="PNG" )
plt.show()


