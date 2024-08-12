import pandas as pd
import numpy as np
import helper
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
fitted=pickle.load(open('fitted.pkl','rb'))
unemployment_df= helper.dataframe()
ny_df_final = unemployment_df['UNRATE'].reset_index()
ny_df_final.columns = ['ds', 'y']
train_data = ny_df_final[ny_df_final['ds'] < '2021-01-01']
test_data = ny_df_final[ny_df_final['ds'] >= '2021-01-01']
train = train_data.set_index('ds')
test = test_data.set_index('ds')
st.title('Unemployment  USA 1970-2024 and Forecast by SARIMA and Fbprophet')
st.plotly_chart(helper.plotly_plot(unemployment_df))
st.write('the numbers is in millions')
n_months = st.sidebar.slider('Number of months to forecast', min_value=1, max_value=120)
if st.sidebar.button('Show Data Analysis'):
    st.header('Seasonal Decomposition Plot')
    result = seasonal_decompose(unemployment_df['UNRATE'], model='multiplicative')
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(unemployment_df['UNRATE'], label='Close')
    # Plot the trend component
    ax.plot(result.trend, label='Trend', color='red')
    # Plot the seasonal component
    ax.plot(result.seasonal, label='Seasonal', color='green')
    # Plot the re
    ax.plot(result.resid, label='Residuals', color='orange')
    ax.set_title('Time Series Decomposition')
    ax.set_xlabel("Time (year)")
    ax.set_ylabel("Closing price")
    ax.legend(loc='best')
    st.pyplot(fig)
    #acf and pcf plot
    st.header('ACF and PCF Plot')
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    plot_acf(unemployment_df['UNRATE'],lags=30, ax=ax[0])
    plot_pacf(unemployment_df['UNRATE'], lags=30, ax=ax[1])
    st.pyplot(fig)
    #Dickey-Fuller Test
    st.header('Dickey-Fuller Test')
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(unemployment_df['UNRATE'])
    st.write('ADF Statistic: ' + str(round(result[0],4)))
    st.write('p-value: '+ str(round(result[1],4)))
    # Fbprophet model stats
    st.header('Fbprophet Model')


    # Test on data from 2021 to 2024
    modelfb=pickle.load(open('fbProfet.pkl','rb'))

    future = modelfb.make_future_dataframe(periods=len(test_data), freq='M')  # MS for monthly, H for hourly
    forecast = modelfb.predict(future)
    fig = modelfb.plot(forecast)
    st.pyplot(fig)
    # Plotting the components
    st.header('The components of the model')
    fig = modelfb.plot_components(forecast)
    st.pyplot(fig)
    # Plotting the forecast
    st.header('The forecast of Prophet over train test data')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_data['ds'], train_data['y'], label='Train')
    ax.plot(test_data['ds'], test_data['y'], label='Test')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Prophet Model - Train, Test, and Forecast')
    st.pyplot(fig)
    # Model Evaluation
    st.header('Model Evaluation')
    from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error
    mse = mean_squared_error(test_data['y'], forecast['yhat'][-len(test_data):])
    mae = mean_absolute_error(test_data['y'], forecast['yhat'][-len(test_data):])
    st.write('Mean Absolute Error: ' + str(round(mae, 4)))
    st.write('Root Mean Squared Error: ' + str(round(np.sqrt(mse), 4)))
    # Sarima model stats
    st.header('Sarima Model')

    # Fit the model

    # Forecast
    pred_test = fitted.predict(start=test.index.min(), end=test.index.max())
    pred_test = pred_test.reset_index()
    # Plotting the forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_data['ds'], train_data['y'], label='Train')
    ax.plot(test_data['ds'], test_data['y'], label='Test')
    ax.plot(pred_test['index'], pred_test['predicted_mean'], label='Forecast')
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('SARIMA Model - Train, Test, and Forecast')
    st.pyplot(fig)
    # Model Evaluation
    from statsmodels.tools.eval_measures import rmse
    st.header('Model Evaluation')
    predictions = pred_test['predicted_mean']
    predictions = predictions[:len(test_data)]
    test_data = test_data[:len(predictions)]
    print("Root Mean Squared Error between actual and  predicted values: ", rmse(predictions, test_data['y']))
    print("Mean Value of Test Dataset:", test_data['y'].mean())
if st.sidebar.button('SARIMA Forecast'):
    st.title('SARIMA Forecast')
    st.write('The SARIMA model is used to forecast the unemployment rate of the USA from 2021 to 2024')
    st.write('enter the number of months you want to forecast')

    # Forecast

    forcast_data = fitted.predict(start=test.index.max(), end=test.index.max() +pd.DateOffset(months=n_months))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_data['ds'], train_data['y'], label='Train')
    ax.plot(test_data['ds'], test_data['y'], label='Test')
    ax.plot(forcast_data.index, forcast_data, label='Forecast future', color='red')
    #ax.plot(pred_test['index'], pred_test['predicted_mean'], label='Forecast')
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('SARIMA Model - Future Forecast')
    st.pyplot(fig)
    st.write('The forecasted values are shown in red')
    forcast_values = forcast_data
    st.header('Forecast at month '+str(n_months) + ' is '+str(round(forcast_values.values[-1],2))+' Million')

if st.sidebar.button('Fbprophet Forecast'):
    st.title('Fbprophet Forecast')
    st.write('The Fbprophet model is used to forecast the unemployment rate of the USA from 2021 to 2024')
    st.write('enter the number of months you want to forecast')

    #model
    modelfb=pickle.load(open('fbProfet_predict_model.pkl','rb'))
    # train and test data
    ny_df_final = unemployment_df['UNRATE'].reset_index()
    ny_df_final.columns = ['ds', 'y']
    train_data = ny_df_final[ny_df_final['ds'] < '2021-01-01']
    # Test on data from 2021 to 2024
    test_data = ny_df_final[ny_df_final['ds'] >= '2021-01-01']
    future = modelfb.make_future_dataframe(periods=+n_months, freq='M')  # MS for monthly, H for hourly
    forecast = modelfb.predict(future)
    fig = modelfb.plot(forecast)
    st.pyplot(fig)
    st.write('The forecasted values are shown in blue')
    # Plotting the forecast
    st.header('The forecast of Prophet Model')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_data['ds'], train_data['y'], label='Train')
    ax.plot(test_data['ds'], test_data['y'], label='Test')
    ax.plot(forecast['ds'].tail(n_months), forecast['yhat'].tail(n_months), label='Forecast')
    ax.legend(loc='best')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Prophet Model -Future Forecast')
    st.pyplot(fig)
    forcast_values = forecast['yhat'].tail(n_months)
    st.header('Forecast at month '+str(n_months) + ' is '+str(round(forcast_values.values[-1],2))+' Million')
if st.sidebar.button('About'):
    st.write('This is a web app that uses the SARIMA and Fbprophet model to forecast the unemployment rate of the USA from 2024 till future')
    st.write('The SARIMA model is used to forecast the unemployment rate of the USA from  2024 till future')
    st.write('The Fbprophet model is used to forecast the unemployment rate of the USA from 2024 till future')
    st.write('The data used is from the US Bureau of Labor Statistics')





























