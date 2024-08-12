
import pandas as pd
import plotly.express as px

import numpy as np
import matplotlib.pyplot as plt
def dataframe():
    unemployment_df=pd.read_csv('unemployment_data.csv')
    unemployment_df['Dates'] = unemployment_df['DATE']
    unemployment_df['DATE'] = pd.to_datetime(unemployment_df['DATE'])
    unemployment_df['Year'] = unemployment_df['DATE'].dt.year
    unemployment_df['Month'] = unemployment_df['DATE'].dt.month
    unemployment_df['day'] = unemployment_df['DATE'].dt.day
    unemployment_df['Name_week'] = unemployment_df['DATE'].dt.day_name()
    unemployment_df.set_index('DATE', inplace=True)
    return unemployment_df
def plotly_plot(unemployment_df):
    fig = px.line(unemployment_df, x='Dates', y='UNRATE', title='Unemployment Rate in USA')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig
def seasonal_decompose_plot(unemployment_df):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(unemployment_df['UNRATE'], model='multiplicative')
    fig, ax = plt.subplots()
    ax.figure(figsize=(16, 8))
    ax.plot(unemployment_df['UNRATE'], label='Close')

    # Plot the trend component
    ax.plot(result.trend, label='Trend', color='red')

    # Plot the seasonal component
    ax.plot(result.seasonal, label='Seasonal', color='green')

    # Plot the re
    ax.plot(result.resid, label='Residuals', color='orange')

    plt.title('Time Series Decomposition')
    plt.xlabel("Time (year)")
    plt.ylabel("Closing price")
    plt.legend(loc='best')
    return fig