---
title: "Time series Forecasting — Random Forrest, ARIMA and Prophet models"
date: 2019-11-07
categories: 
  - Data Science
tags: [machine learning, time series]
excerpt: "Compare ARIMA, Prophet and Random Forrest for forecast of production amounts"
---
## Introduction


Forecasting tool for production amounts of a German food producer. Three forecasting models will be compared for this Time Series Forecasting:
- [Random Forrest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) with features of average historic amounts and German holidays
- [Seasonal ARIMA (SARIMA)](https://medium.com/analytics-vidhya/sarima-forecasting-seasonal-data-with-python-and-r-2e7472dfad83)
- [Facebook Prophet](https://facebook.github.io/prophet)


## Import libraries


```python
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
```

## Read csv files


```python
#Amounts 2017
posten17 = pd.read_csv('2017_Auftragsposten.csv')
#Amounts 2018
posten18= pd.read_csv('2018_Auftragsposten.csv')
#Amounts 2019
posten19 = pd.read_csv('2019_Auftragsposten.csv')
posten19.head()
#Holidays
holidays = pd.read_csv('Feiertage.csv')
```

## Clean and aggreate data


```python
# combine the files
posten = pd.concat([posten17,posten18,posten19],sort=False)

#delete AKTION
posten.drop(posten[posten.Belegtypencode == 'AKTION'].index, inplace=True)

#rename columns
posten.rename(columns={'Kunde': 'cust',
                                'Produktionsdatum': 'date',
                                'Produkt': 'prod',
                                'Schale': 'bowl',
                                'Menge (Basis)': 'amount',
                                'Nettogewicht': 'weight'}, inplace=True)

#Apply unique name for 8ECK
posten.loc[posten['bowl'] == '8eck', 'bowl'] = '8ECK'

#datetime for date
posten.date = pd.to_datetime(posten.date)
holidays.date = pd.to_datetime(holidays.date)

#prepare holidays
holidays.set_index('date',inplace=True)
holidays = holidays.hol.resample('W').sum().to_frame()

#Aggregate over date and product or date and bowl
prods = posten.groupby(['date','prod'])[['weight']].agg('sum')
bowls = posten.groupby(['date','bowl'])[['amount']].agg('sum') 

#Put index back in columns
bowls = bowls.reset_index('bowl')
prods = prods.reset_index('prod')

#Split into different products/bowls
gn = bowls[bowls.bowl == 'GN 1/4']['amount'].resample('W').sum().to_frame()
eck = bowls[bowls.bowl == '8ECK']['amount'].resample('W').sum().to_frame()
prods = prods[prods['prod'] == 'FP GEFÜLLT']['weight'].resample('W').sum().to_frame()
```

## Visualitazion of Production Data


```python
ax = gn.amount.plot(label='GN 1/4', figsize=(15, 7))
ax.set_xlabel('Date')
ax.set_ylabel('Amount per week')
plt.legend()
plt.show()

ax = eck.amount.plot(label='8ECK', figsize=(15, 7))
ax.set_xlabel('Date')
ax.set_ylabel('Amount per week')
plt.legend()
plt.show()

ax = prods.weight.plot(label='FP UNGEFUELLT', figsize=(15, 7))
ax.set_xlabel('Date')
ax.set_ylabel('Weight week')
plt.legend()
plt.show()
```


![png]({{ '/assets/images/2019-11-07/output_10_0.png' }})



![png]({{ '/assets/images/2019-11-07/output_10_1.png' }})



![png]({{ '/assets/images/2019-11-07/output_10_2.png' }})


## Random Forrest Regressor
First add new features of historic amounts and German holidays:
- Amount one year ago
- Average amount over last 6,8 and 12 weeks
- Week number
- Year number
- Number of holidays current and following week

Then train Random Forrest Regressor with these features. At the end I calculate the errors on cross validation data and compare it with ther error for just forcasting based on the last 6 weeks average amount. This will be done for three different products.

Function for creating additional features and training the model


```python
#Function to calculate average ammounts
def process_df(df_raw):
    """returns dataframe with average historic amounts and holidays
    """
    df = df_raw.copy()
    df['1year'] = np.nan
    df['12weeks'] = np.nan
    df['8weeks'] = np.nan
    df['6weeks'] = np.nan
    df['week'] = df.index.week
    df['year'] = df.index.year
    for i in range(0,len(df.index)):
        df.iloc[i,1] = df.iloc[i-52,0] if (i>=52) else df.iloc[i-12:i,0].mean()
        df.iloc[i,2] = df.iloc[i-12:i,0].mean()
        df.iloc[i,3] = df.iloc[i-8:i,0].mean()
        df.iloc[i,4] = df.iloc[i-6:i,0].mean()
    #combine with holidays   
    df = pd.concat([df,holidays],axis=1)
    #holidays in next week
    df['hol_next'] = df.hol.shift(-1) 
    return df 

from sklearn.metrics import mean_absolute_error,mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

#Function to train model and print error metrics
def create_model(df,date):
    test_data = df.copy()
    test_data.dropna(axis=0,inplace=True)

    #select prediction target
    y = test_data.ix[:,0]

    #Features
    features = ['1year','12weeks','8weeks','6weeks','week','year','hol','hol_next']
    X = test_data[features]

    # Split into validation and training data
    #train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=2)
    train_X = X[:date]
    train_y = y[:date]
    val_X = X[date:]
    val_y = y[date:]

    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(n_estimators=160,random_state=1)

    # fit your models
    rf_model.fit(train_X, train_y)

    # Calculate the mean squared log error of your Random Forest model on the validation data
    rf_val_msle = mean_squared_log_error(val_y, rf_model.predict(val_X))
    rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))

    # Calculate errors on train data
    rf_train_msle = mean_squared_log_error(train_y, rf_model.predict(train_X))
    rf_train_mae = mean_absolute_error(train_y, rf_model.predict(train_X))

    # Calculate mean error if you just take 6 week average
    weekAvg_msle = mean_squared_log_error(val_y, val_X['6weeks'])
    weekAvg_mae = mean_absolute_error(val_y, val_X['6weeks'])

    #print(f'Mean squared log error for test data: {rf_train_msle}')
    #print(f'Mean average error for test data: {rf_train_mae}')
    print(f'Mean squared log error for Validation data: {rf_val_msle}')
    print(f'Mean average error for Validation data: {rf_val_mae}')
    print(f'Mean squared log error for just taking last 6 weeks: {weekAvg_msle}')
    print(f'Mean average error for just taking last 6 weeks: {weekAvg_mae}')

    return rf_model

```

Run this for different products


```python
gn_processed = process_df(gn)
eck_processed = process_df(eck)
prods_processed = process_df(prods)
 
print('FP Ungefuellt')
rf_model_prods = create_model(prods_processed,'2019-06-02')
print('GN 1/4')
rf_model_gn = create_model(gn_processed,'2019-06-02')
print('8 Eck')
rf_model_eck = create_model(eck_processed,'2019-06-02')
```

    FP Ungefuellt
    Mean squared log error for Validation data: 0.01898634528409573
    Mean average error for Validation data: 3331.9344776785865
    Mean squared log error for just taking last 6 weeks: 0.04738173222656172
    Mean average error for just taking last 6 weeks: 4604.807142857143
    GN 1/4
    Mean squared log error for Validation data: 0.01034894718182547
    Mean average error for Validation data: 2517.8736607142855
    Mean squared log error for just taking last 6 weeks: 0.03367406906924567
    Mean average error for just taking last 6 weeks: 3934.511904761904
    8 Eck
    Mean squared log error for Validation data: 0.08487610454676411
    Mean average error for Validation data: 23944.734375000004
    Mean squared log error for just taking last 6 weeks: 0.10915466774421032
    Mean average error for just taking last 6 weeks: 28685.452380952378
    

## Forecast mit Random Forrest

Calculate/get all features, then call prediction with Random Forrest model


```python
def predict(weeks,df,model):
    df = df.dropna(axis=0)
    last_date = df.iloc[[-1]].index
    for i in range(0,weeks):
        last_date = last_date + pd.DateOffset(days=7)
        next_date = last_date + pd.DateOffset(days=7)
        df = df.append(pd.DataFrame(index=[last_date[0]]),sort=False)
        df.iloc[-1,1] = df.iloc[-53,0]
        df.iloc[-1,2] = df.iloc[-13:-1,0].mean()
        df.iloc[-1,3] = df.iloc[-9:-1,0].mean()
        df.iloc[-1,4] = df.iloc[-7:-1,0].mean()
        df.iloc[-1,5] = last_date.week[0]
        df.iloc[-1,6] = df.iloc[[-1]].index.year[0]
        df.iloc[-1,7] = holidays.loc[last_date].hol[0]
        df.iloc[-1,8] = holidays.loc[next_date].hol[0]
        df.iloc[-1,0] = model.predict([df.iloc[-1,1:]])[0]   
    return df
     
eck_predict_rf = predict(52,eck_processed[:'2018'],rf_model_eck)
gn_predict_rf = predict(52,gn_processed[:'2018'],rf_model_gn)
prods_predict_rf = predict(52,prods_processed[:'2018'],rf_model_prods)

```

## Forecast with ARIMA

Get best parameters for ARIMA (with lowest AIC)


```python
y = prods[:'2018'].ix[:,0]
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
for param in pdq:
    for param_seasonal in seasonal_pdq:
      try: 
        mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
        results = mod.fit()
        print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
      except: continue
```

    ARIMA(0, 0, 0)x(0, 0, 0, 52) - AIC:2372.1259213250455
    ARIMA(0, 0, 0)x(0, 1, 0, 52) - AIC:1101.482494779482
    ARIMA(0, 0, 0)x(1, 0, 0, 52) - AIC:1074.7508397031318
    ARIMA(0, 0, 1)x(0, 0, 0, 52) - AIC:2327.4187618024257
    ARIMA(0, 0, 1)x(0, 1, 0, 52) - AIC:1058.4779225184222
    ARIMA(0, 0, 1)x(1, 0, 0, 52) - AIC:1069.7766642743488
    ARIMA(0, 1, 0)x(0, 0, 0, 52) - AIC:2111.460792739303
    ARIMA(0, 1, 0)x(0, 1, 0, 52) - AIC:1041.821902974718
    ARIMA(0, 1, 0)x(1, 0, 0, 52) - AIC:1064.1914130561297
    ARIMA(0, 1, 1)x(0, 0, 0, 52) - AIC:2049.2781207771195
    ARIMA(0, 1, 1)x(0, 1, 0, 52) - AIC:1008.6104812945237
    ARIMA(0, 1, 1)x(1, 0, 0, 52) - AIC:1055.704038430932
    ARIMA(1, 0, 0)x(0, 0, 0, 52) - AIC:2130.8094443731925
    ARIMA(1, 0, 0)x(0, 1, 0, 52) - AIC:1057.5715840950882
    ARIMA(1, 0, 0)x(1, 0, 0, 52) - AIC:1050.2564821973467
    ARIMA(1, 0, 1)x(0, 0, 0, 52) - AIC:2067.3872084395152
    ARIMA(1, 0, 1)x(0, 1, 0, 52) - AIC:1032.3755623684285
    ARIMA(1, 0, 1)x(1, 0, 0, 52) - AIC:1051.8883547043051
    ARIMA(1, 1, 0)x(0, 0, 0, 52) - AIC:2093.246568673168
    ARIMA(1, 1, 0)x(0, 1, 0, 52) - AIC:1038.0225727262589
    ARIMA(1, 1, 0)x(1, 0, 0, 52) - AIC:1040.6054336927996
    ARIMA(1, 1, 1)x(0, 0, 0, 52) - AIC:2051.253658146207
    ARIMA(1, 1, 1)x(0, 1, 0, 52) - AIC:1009.7395878207528
    ARIMA(1, 1, 1)x(1, 0, 0, 52) - AIC:1031.721189448723
    

Then train model and predict.


```python
def predict_arima(df,date,order,seasonal_order):
    y = df[:date].ix[:,0]
    mod = sm.tsa.statespace.SARIMAX(y,order=order,seasonal_order=seasonal_order,enforce_stationarity=False,enforce_invertibility=False)
    results = mod.fit()
    return results.get_forecast(steps=52)

eck_predict_arima = predict_arima(eck,'2018',(1, 1, 1),(0, 1, 0, 52))
gn_predict_arima = predict_arima(gn,'2018',(1, 1, 1),(0, 1, 0, 52))
prods_predict_arima = predict_arima(prods,'2018',(1, 1, 1),(0, 1, 0, 52))

```

## Forecast with Prophet


```python
from fbprophet import Prophet
def predict_prophet(df,date):
    train_data = df.copy()
    train_data = train_data[:date]
    train_data = train_data.reset_index()
    train_data.columns = ['ds','y'] # To use prophet column names should be like that
    m = Prophet(yearly_seasonality=True,weekly_seasonality=True)
    m.fit(train_data)
    future = m.make_future_dataframe(periods=30,freq='W')
    prophet_pred = m.predict(future)
    prophet_pred = pd.DataFrame({"Date" : prophet_pred['ds'], "Pred" : prophet_pred["yhat"]})
    prophet_pred = prophet_pred.set_index("Date")
    prophet_pred.index.freq = "W"
    return prophet_pred

eck_predict_prophet = predict_prophet(eck,'2019-06-02')
gn_predict_prophet = predict_prophet(gn,'2019-06-02')
prods_predict_prophet = predict_prophet(prods,'2019-06-02')
```

    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    

## Compare Models
Calculate "mean absolute error" und "mean squared log error" of the different models and run it for different products


```python
def compare_models(val,predict_rf,predict_prophet,predict_arima):
    arima_msle = mean_squared_log_error(val['2019':'2019-09-01'].ix[:,0], predict_arima.predicted_mean['2019':'2019-09-01'])
    arima_mae = mean_absolute_error(val['2019':'2019-09-01'].ix[:,0], predict_arima.predicted_mean['2019':'2019-09-01'])
    rf_msle = mean_squared_log_error(val['2019':'2019-09-01'].ix[:,0], predict_rf['2019':'2019-09-01'].ix[:,0])
    rf_mae = mean_absolute_error(gn['2019':'2019-09-01'], predict_rf['2019':'2019-09-01'].ix[:,0])
    prophet_msle = mean_squared_log_error(gn['2019':'2019-09-01'], predict_prophet['2019':'2019-09-01'].Pred)
    prophet_mae = mean_absolute_error(gn['2019':'2019-09-01'], predict_prophet['2019':'2019-09-01'].Pred)

    print(f'Squared log error ARIMA: {arima_msle}')
    print(f'Mean average error ARIMA: {arima_mae}')
    print(f'Squared log error RF: {rf_msle}')
    print(f'Mean average error RF: {rf_mae}')
    print(f'Squared log error Prophet: {prophet_msle}')
    print(f'Mean average error Prophet: {prophet_mae}')

print('GN 1/4')
compare_models(gn,gn_predict_rf,gn_predict_prophet,gn_predict_arima)
print('8ECK')
compare_models(prods,eck_predict_rf,eck_predict_prophet,eck_predict_arima)
print('FP Ungefuellt')
compare_models(prods,prods_predict_rf,prods_predict_prophet,prods_predict_arima)

```

    GN 1/4
    Squared log error ARIMA: 0.031319039610705195
    Mean average error ARIMA: 4393.944903417675
    Squared log error RF: 0.007924204521299077
    Mean average error RF: 2138.39625
    Squared log error Prophet: 0.022212620724342
    Mean average error Prophet: 3599.760811242864
    8ECK
    Squared log error ARIMA: 4.246678205516045
    Mean average error ARIMA: 190905.12246113332
    Squared log error RF: 2.5644360352216062
    Mean average error RF: 103898.77928571429
    Squared log error Prophet: 2.7946640870325266
    Mean average error Prophet: 143134.97627760138
    FP Ungefuellt
    Squared log error ARIMA: 0.11273534546531086
    Mean average error ARIMA: 8440.78315007046
    Squared log error RF: 0.01405137744987062
    Mean average error RF: 6181.318673214287
    Squared log error Prophet: 0.029611387763729846
    Mean average error Prophet: 4207.992366169594
    

## Visualization with forecast


```python
pd.plotting.register_matplotlib_converters()
ax = prods['2019-01-06':].ix[:,0].plot(label='observed')
prods_predict_rf['2019-01-06':].ix[:,0].plot(ax=ax, label='forecast RF', alpha=.7, figsize=(15, 7))
prods_predict_prophet['2019-01-06':].ix[:,0].plot(ax=ax, label='forecast Prophet', alpha=.7, figsize=(15, 7))
prods_predict_arima.predicted_mean.plot(ax=ax, label='forecast ARIMA', alpha=.7, figsize=(15, 7))
ax.set_xlabel('Date')
ax.set_ylabel('Weight')
ax.set_title('FP UNGEFUELLT')
plt.legend()
plt.show()

ax = gn['2019-01-06':].ix[:,0].plot(label='observed')
gn_predict_rf['2019-01-06':].ix[:,0].plot(ax=ax, label='forecast RF', alpha=.7, figsize=(15, 7))
gn_predict_prophet['2019-01-06':].ix[:,0].plot(ax=ax, label='forecast Prophet', alpha=.7, figsize=(15, 7))
gn_predict_arima.predicted_mean.plot(ax=ax, label='forecast ARIMA', alpha=.7, figsize=(15, 7))
ax.set_xlabel('Date')
ax.set_ylabel('Amount')
ax.set_title('GN 1/4')
plt.legend()
plt.show()

ax = eck['2019-01-06':].ix[:,0].plot(label='observed')
eck_predict_rf['2019-01-06':].ix[:,0].plot(ax=ax, label='forecast RF', alpha=.7, figsize=(15, 7))
eck_predict_prophet['2019-01-06':].ix[:,0].plot(ax=ax, label='forecast Prophet', alpha=.7, figsize=(15, 7))
eck_predict_arima.predicted_mean.plot(ax=ax, label='forecast ARIMA', alpha=.7, figsize=(15, 7))
ax.set_xlabel('Date')
ax.set_ylabel('Amount')
ax.set_title('8ECK')
plt.legend()
plt.show()
```


![png]({{ '/assets/images/2019-11-07/output_29_0.png' }})



![png]({{ '/assets/images/2019-11-07/output_29_1.png' }})



![png]({{ '/assets/images/2019-11-07/output_29_2.png' }})

