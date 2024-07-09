
# root/data_processing/time_series_analyzer.py
# Implements tools for analyzing time series data

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt

class TimeSeriesAnalyzer:
    def __init__(self):
        pass

    def check_stationarity(self, timeseries):
        result = adfuller(timeseries, autolag='AIC')
        return {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }

    def decompose_series(self, timeseries, model='additive', period=None):
        if period is None:
            period = len(timeseries) // 7  # Assume weekly seasonality if not specified
        result = seasonal_decompose(timeseries, model=model, period=period)
        return result

    def fit_arima(self, timeseries, order=(1,1,1)):
        model = ARIMA(timeseries, order=order)
        results = model.fit()
        return results

    def fit_prophet(self, df):
        model = Prophet()
        model.fit(df)
        return model

    def analyze_timeseries(self, data, date_column, value_column):
        df = pd.DataFrame({
            'ds': pd.to_datetime(data[date_column]),
            'y': data[value_column]
        })

        # Check stationarity
        stationarity = self.check_stationarity(df['y'])

        # Decompose series
        decomposition = self.decompose_series(df.set_index('ds')['y'])

        # Fit ARIMA model
        arima_model = self.fit_arima(df['y'])

        # Fit Prophet model
        prophet_model = self.fit_prophet(df)

        # Make future predictions with Prophet
        future = prophet_model.make_future_dataframe(periods=365)
        forecast = prophet_model.predict(future)

        return {
            'stationarity': stationarity,
            'decomposition': decomposition,
            'arima_model': arima_model,
            'prophet_forecast': forecast
        }

if __name__ == '__main__':
    analyzer = TimeSeriesAnalyzer()
    # Example usage (you would load your own data here)
    data = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=1000),
        'value': np.random.randn(1000).cumsum()
    })
    results = analyzer.analyze_timeseries(data, 'date', 'value')
    print(results['stationarity'])
    results['decomposition'].plot()
    plt.show()
