import streamlit as st 

import matplotlib.pyplot as plt

import pandas as pd
from pandas import read_csv
from datetime import datetime

from sklearn.preprocessing import Normalizer, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler  
from sklearn.compose import make_column_transformer 

from sklearn.model_selection import train_test_split
import requests
import numpy as np
from sklearn import metrics

from sodapy import Socrata

from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor


@st.cache_data  # 👈 Add the caching decorator
def load_data(url, sep):
    df = pd.read_csv(url, sep)
    return df

# CACHE API CALLS
# @st.cache_data
# def api_call():
#     response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
#     return response.json()


df = load_data("C:\\Users\\alexm\\OneDrive\\Projects\\PV Energy\\Calgary_Dataset.csv", sep = '\t')


panels = set(np.array(df['name']))


st.title('Predicting Energy Output for Solar Panels')

st.write("""
# Explore different substations
Which one is the best?
""")

substation_name = st.sidebar.selectbox(
    'Select Substation',
    panels
)

st.write(f"## {substation_name} Dataset")

param_list = ['temperature_2m', 'relativehumidity_2m', 'apparent_temperature', 'dewpoint_2m', 'surface_pressure', 'snowfall', 'weathercode', 'precipitation', 'cloudcover',
              'direct_radiation', 'direct_normal_irradiance', 'diffuse_radiation', 'windspeed_10m', 'winddirection_10m', 'rain', 'surface_pressure', 'shortwave_radiation', 'vapor_pressure_deficit', 
              'soil_temperature_7_to_28cm', 'soil_moisture_7_to_28cm']


station_df = df.loc[df['name'] == substation_name]
@st.cache_data
def percentage(x):  
    x = round(x*100,2)
    return (str(x) + "%")

@st.cache_data
def make_regression_ann(initializer='normal', activation='relu', optimizer='adam', loss='mse'):

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=40, input_dim=19, kernel_initializer=initializer, activation=activation))
    model.add(keras.layers.Dense(units=40, kernel_initializer=initializer, activation=activation))
    #model.add(keras.layers.Dense(units=40, kernel_initializer=initializer, activation=activation))
    model.add(keras.layers.Dense(1, kernel_initializer=initializer, activation = 'linear'))
    model.compile(loss=loss, optimizer=optimizer)

    return model

TargetVariable=['kwh']
Predictors=[x for x in param_list if x != 'weathercode']
Categorical_Variables = ['weathercode', 'name']

@st.cache_data
def standardize_and_split(df):
    # Separate Target Variable and Predictor Variables

    ### Standardization of data ###

    X=station_df[Predictors]
    y=station_df[TargetVariable]

    PredictorScaler=StandardScaler()
    TargetVarScaler=StandardScaler()
    
    # Storing the fit object for later reference
    PredictorScalerFit = PredictorScaler.fit(X)
    TargetVarScalerFit = TargetVarScaler.fit(y)
    
    # Generating the standardized values of X and y
    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = standardize_and_split(station_df)



#preds = grid_search.predict(X_test)

#use the best parameters found by the grid search
model = KerasRegressor(make_regression_ann, batch_size= 128, epochs= 75, optimizer='adam')


md = model.fit(X_train, y_train, verbose=0)
preds = md.predict(X_test)

TestingData=pd.DataFrame(data=X_test.copy(), columns=station_df[Predictors].columns)
TestingData['TotalPower']=y_test
TestingData['PredictedTotalPower']=preds
TestingData.head()


mae = metrics.mean_absolute_error(TestingData['TotalPower'], TestingData['PredictedTotalPower'])
mse = metrics.mean_squared_error(TestingData['TotalPower'], TestingData['PredictedTotalPower'])
rmse = np.sqrt(mse) # or mse**(0.5)  
r2 = metrics.r2_score(TestingData['TotalPower'],TestingData['PredictedTotalPower'])



st.write("# Results of sklearn.metrics:")

col1, col2, col3, col4 = st.columns(4)

col1.metric(":green[**MAE**]",round(mae,2))
col2.metric(":green[**MSE**]", round(mse,2))
col3.metric(":green[**RMSE**]", round(rmse,2))
col4.metric(":green[**R_Square**]", percentage(r2))


#cd to current folder -> streamlit run dash.py
