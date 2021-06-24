import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf,pacf
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#................. Creating Folder for saving the Figures.......................
if not os.path.exists('Time_series_results'):
    os.makedirs('Time_series_results')
#...........Reading the data..................

data = pd.read_excel('Bases_Final_ADS_Jun2021.xlsx')

#....................... seperating Training data...................
mask = (data['data_tidy'] >= '2004-01-01') & (data['data_tidy'] <= '2021-02-01')
mask2 = (data['data_tidy'] >= '2021-03-01')
dt = data.loc[mask]
prediction_dt = data.loc[mask2]

#.................... Filling NULL values..................................
renda_mean, massa_mean = dt['renda_r'].mean(), dt['massa_r'].mean()
dt['renda_r'].fillna(value=renda_mean, inplace=True)
dt['massa_r'].fillna(value=massa_mean, inplace=True)

#...............................Selecting Southeast Vaariable/Features...........................................

ind_se_dt = dt[['data_tidy', 'ind_se','renda_r', 'pop_ocup_br', 'massa_r', 'du','pmc_a_se',
        'temp_max_se', 'temp_min_se', 'pmc_r_se', 'pim_se']]

ind_se_dt.set_index('data_tidy', inplace=True) #set date as index

time_series_ind_se =ind_se_dt[['ind_se']]    

#>>>>>>............... Visualisation..............................................
plt.figure(figsize=(10,5))
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Consumption (Gwh)", fontsize = 12)
plt.title("Industrial Energy Consumption Southeast", fontsize = 15)
plt.plot(time_series_ind_se)
plt.xticks( fontsize = 12)
plt.yticks( fontsize = 12)
plt.savefig('Time_series_results/Industrial_Energy_Consumption_Southeast' + '.png', format="PNG" )
plt.show()

###.......................Seasonal Decompose................................
componenets = seasonal_decompose(time_series_ind_se, model='multiplicative')
z = componenets.plot()
z.savefig('Time_series_results/Decompose_componenets_series' + '.png', format="PNG" )
plt.show()

###................ Stationary test and Rolling mean.........................................

rolmean = time_series_ind_se.rolling(12).mean()   
rolstd = time_series_ind_se.rolling(12).std()

plt.plot(time_series_ind_se, color='blue',label='Original')
plt.plot(rolmean, color='red', label='Rolling Mean')
plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Consumption (Gwh)", fontsize = 12)
plt.savefig('Time_series_results/Rolling_Mean_Standard Deviation' + '.png', format="PNG" )
plt.show()

#........................ADF test for stationarity...............................
adft = adfuller(time_series_ind_se['ind_se'],autolag='AIC')
adft_output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
for key,values in adft[4].items():
        adft_output['critical value (%s)'%key] =  values

adft_output.to_csv('Time_series_results/dickey_fuller_test_before_smoothing.csv')
print(adft_output)

####......................Smooting using Log.............................................
df_log = np.log(time_series_ind_se)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.plot(df_log, color='blue',label='Original')
plt.plot(moving_avg, color="red", label = 'Rolling mean')
plt.plot(std_dev, color ="black", label = 'Rolling std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation of Log values')
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Consumption (Gwh) log values", fontsize = 12)

plt.savefig('Time_series_results/Rolling_Mean_Standard_Deviation_Log_Values' + '.png', format="PNG" )
plt.show()

#............... Difference Log ANd Moving average..................................
df_log_moving_avg_diff = df_log - moving_avg
df_log_moving_avg_diff.dropna(inplace=True)

df_log_moving_avg_mean = df_log_moving_avg_diff.rolling(12).mean()
df_log_moving_avg_std_dev = df_log.rolling(12).std()
plt.plot(df_log_moving_avg_diff, color='blue',label='log - moving avg.')
plt.plot(df_log_moving_avg_mean, color="red", label = 'Rolling mean')
plt.plot(df_log_moving_avg_std_dev, color ="black", label = 'Rolling std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation of Difference')
plt.xlabel("Years", fontsize = 12)
plt.savefig('Time_series_results/Rolling_Mean_Standard_Deviation_Log_Diffeence_Values' + '.png', format="PNG" )
plt.show()

#........................ADF test for stationarity...............................
adft = adfuller(df_log_moving_avg_diff['ind_se'],autolag='AIC')
adft_output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
for key,values in adft[4].items():
    adft_output['critical value (%s)'%key] =  values

adft_output.to_csv('Time_series_results/dickey_fuller_test_after_smoothing.csv')
print(adft_output)

###......................Integrated Differencing...................................

Integrat_diff = df_log - df_log.shift()
plt.title("Shifted timeseries")
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Consumption", fontsize = 12)
plt.plot(Integrat_diff)#Let us test the stationarity of our resultant series
plt.savefig('Time_series_results/Shifted_timeseries' + '.png', format="PNG" )
plt.show()

Integrat_diff.dropna(inplace=True)

Integrat_diff_mean = Integrat_diff.rolling(12).mean()
Integrat_diff_std_dev = Integrat_diff.rolling(12).std()
plt.plot(Integrat_diff, label = 'Differencing')
plt.plot(Integrat_diff_mean, color="red", label = 'Rolling mean')
plt.plot(Integrat_diff_std_dev, color ="black", label = 'Rolling std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation of integrated Differencing')
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Consumption", fontsize = 12)
plt.legend(loc='best')
plt.savefig('Time_series_results/Rolling_Mean_Standard_Deviation_Log_Integrat_differencing' + '.png', format="PNG" )
plt.show()

#........................ADF test for stationarity...............................
adft = adfuller(Integrat_diff['ind_se'],autolag='AIC')
adft_output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
for key,values in adft[4].items():
    adft_output['critical value (%s)'%key] =  values

adft_output.to_csv('Time_series_results/dickey_fuller_test_after_Differencing.csv')
print(adft_output)

###..................Decomposition................................................
result = seasonal_decompose(df_log, model='additive', freq = 12)

trend = result.trend
trend.dropna(inplace=True)
seasonality = result.seasonal
seasonality.dropna(inplace=True)
residual = result.resid
residual.dropna(inplace=True)

residual_mean = residual.rolling(12).mean()
residual_std_dev = residual.rolling(12).std()
plt.plot(residual, label = 'Residual')
plt.plot(residual_mean, color="red", label = 'Rolling mean')
plt.plot(residual_std_dev, color ="black", label = 'Rolling std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation of Residual')
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Consumption", fontsize = 12)
plt.savefig('Time_series_results/Rolling_Mean_Standard_Deviation_residual' + '.png', format="PNG" )
plt.show()

#........................ADF test for stationarity...............................
adft = adfuller(residual['ind_se'],autolag='AIC')
adft_output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
for key,values in adft[4].items():
    adft_output['critical value (%s)'%key] =  values

adft_output.to_csv('Time_series_results/dickey_fuller_test_after_residual.csv')
print(adft_output)

#..........Auto Correlation ACF and PACF....................

acf = acf(Integrat_diff, nlags=15)
pacf= pacf(Integrat_diff, nlags=15,method='ols')
plt.subplot(121)
plt.plot(acf) 
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(Integrat_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(Integrat_diff)),linestyle='--',color='black')
plt.title('Auto corellation function')
plt.tight_layout()

plt.subplot(122)
plt.plot(pacf) 
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(Integrat_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(Integrat_diff)),linestyle='--',color='black')
plt.title('Partially auto corellation function')
plt.tight_layout()
plt.savefig('Time_series_results/ACF_PACF' + '.png', format="PNG" )
plt.show()

#....................ARIMA Model..............................................
model = ARIMA(df_log, order=(1,1,1))
result_AR = model.fit()
plt.plot(Integrat_diff)
plt.plot(result_AR.fittedvalues, color='red')
plt.title("sum of squares of residuals")
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Consumption", fontsize = 12)
plt.savefig('Time_series_results/sum_squares_residuals' + '.png', format="PNG" )
plt.show()

#print('RMSE : %f' %np.sqrt((sum((result_AR.fittedvalues- Integrat_diff["ind_se"])**2))/len(Integrat_diff)))
print('RMSE : %f' %np.sqrt(sum((result_AR.fittedvalues - Integrat_diff["ind_se"])**2)/len(Integrat_diff)))
print('MSE : %f' %(sum((result_AR.fittedvalues - Integrat_diff["ind_se"])**2)/len(Integrat_diff)))
print('R2-score : %f' %r2_score(Integrat_diff["ind_se"] , result_AR.fittedvalues ))

res = result_AR.plot_predict(2,230)

res.savefig('Time_series_results/prediction' + '.png', format="PNG" )

plt.show()

