import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import shap
import time
import joblib
import warnings
import pickle
from urllib.request import urlopen
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#disable pointless tensorflow warnings
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def load_data(url, sep):
    df = pd.read_csv(url, sep, parse_dates=['timestamp'], index_col='timestamp')
    return df

# Define constants in a dictionary
config = {
    'file_path': "https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/SunPower_Full.csv",
    'target_variable': 'Active_Power',
    'predictors': ['temperature_2m', 'relativehumidity_2m', 'direct_radiation', 'diffuse_radiation',  'windspeed_10m', 'cloudcover', 'season'],
    'categorical_variables': ['season'],
    'time_intervals': ['first_interval','second_interval','third_interval','fourth_interval','fifth_interval','sixth_interval'],
    'weather_types': ['TypeA', 'TypeB', 'TypeC'],
    'standardize_predictor_list': ['temperature_2m', 'relativehumidity_2m', 'direct_radiation', 'diffuse_radiation',  'windspeed_10m', 'cloudcover']
}


# Load data
def load_data(file_path):
    df = pd.read_csv(file_path, sep = '\t')
    df.rename(columns = {'timestamp':'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df[config['target_variable']] = df[config['target_variable']].clip(lower=0)   #set negative values to 0 
    return df

# Add season
def add_season(df):
    def season(month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    df['season'] = df['date'].dt.month.apply(season)
    return df

# Choose only 7-18 interval
def choose_interval(df):
    df = df.sort_values('date')
    df = df.set_index('date')
    df = df.between_time('07:00', '18:00')
    return df

# Split data
def split_data(df):
    ord_enc = OrdinalEncoder()
    season = ord_enc.fit_transform(np.array(df['season']).reshape(-1,1))
    df['season'] = season
    cutoff_date = df.index.min() + pd.DateOffset(years=7)   
    train = df.loc[:cutoff_date]
    test = df.loc[cutoff_date+ pd.DateOffset(hours=1):]
    return train, test

# Detect time interval
def detect_time_interval(df):
    df_time_detect = df.copy()
    intervals = {'first_interval': (7, 9), 'second_interval': (9, 11), 'third_interval': (11, 13),
                'fourth_interval': (13, 15), 'fifth_interval': (15, 17), 'sixth_interval': (17, 18)}
    df_time_detect['time_interval'] = pd.cut(df_time_detect.index.hour, bins=[interval[0] for interval in intervals.values()] + [24],
                                labels=[interval_name for interval_name in intervals.keys()],
                                include_lowest=True, right=False)
    return df_time_detect

# Create weather type
def create_weather_type(train):
    new_train = pd.DataFrame()
    for interval in config['time_intervals']:
        train_df = train[train['time_interval'] == interval].copy()
        weather_type = []
        avg_kwh = np.mean(train_df[config['target_variable']])
        max_kwh = max(train_df[config['target_variable']])
        min_kwh = min(train_df[config['target_variable']])
        for y in train_df[config['target_variable']]:
            if (y >= avg_kwh):
                weather_type.append("TypeA")
            elif (y > avg_kwh - (avg_kwh - min_kwh) / 2 ) and (y <  avg_kwh):
                weather_type.append("TypeB")
            elif (y >= 0) and (y<= avg_kwh - (avg_kwh - min_kwh)/2):
                weather_type.append("TypeC")
            else:
                raise ValueError(f"Something wrong happened in weather type classification for {interval}")
        train_df['weather_type'] = weather_type
        new_train = pd.concat([new_train, train_df])
    new_train = new_train.sort_index()
    return new_train

# Train random forest classifier
def train_rf_classifier(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rfc = RandomForestClassifier()
    grid_search = GridSearchCV(rfc, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search

# Predict weather type
def predict_weather_type(grid_search, X_test):
    y_pred = grid_search.best_estimator_.predict(X_test)
    X_test.loc[:,'weather_type'] = y_pred
    return X_test

# Classify weather type
def classify_weather_type(new_train, test):
    new_test = pd.DataFrame()
    for interval in config['time_intervals']:
        interval_train_dataset = new_train[new_train['time_interval'] == interval].copy()
        interval_test_dataset = test[test['time_interval'] == interval].copy()
        try:
            grid = joblib.load(urlopen(f'https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/ClassifiedWeatherTypes/RF_Weather_{interval}_.pkl'))
            #grid = joblib.load(f'ClassifiedWeatherTypes\\RF_Weather_{interval}_.pkl')  #load fitted model if exists
            classified_weather_type = predict_weather_type(grid, interval_test_dataset[config['predictors']].copy())
        except:
            grid = train_rf_classifier(interval_train_dataset[config['predictors']] , interval_train_dataset['weather_type'])
            joblib.dump(grid, f'ClassifiedWeatherTypes\\RF_Weather_{interval}_.pkl') #save fitted model
            classified_weather_type = predict_weather_type(grid, interval_test_dataset[config['predictors']].copy())
        print(f"Grid Search training done for {interval}")
        classified_weather_type = predict_weather_type(grid, interval_test_dataset[config['predictors']].copy())
        classified_weather_type['time_interval'] = interval
        print(f"Weather type Predictions done for {interval}")
        new_test = pd.concat([new_test, classified_weather_type])
    new_test = new_test.sort_index()
    return new_test

# Standardize data
def standardize_data(new_train, new_test):
    X_new_train = new_train[config['standardize_predictor_list']]
    X_new_test = new_test[config['standardize_predictor_list']]
    predictor_scaler = StandardScaler()
    predictor_scaler_fit = predictor_scaler.fit(X_new_train)
    #save fitted predictor
    try:
        with open('Fitted_Standardizers\\std_scaler.bin', 'wb') as f:
            joblib.dump(predictor_scaler_fit, f)
    except:
        pass
    X_new_train= predictor_scaler_fit.transform(X_new_train)
    X_new_test = predictor_scaler_fit.transform(X_new_test)
    new_stand_train = pd.DataFrame(X_new_train, index=new_train[config['standardize_predictor_list']].index, columns=new_train[config['standardize_predictor_list']].columns)
    new_stand_test = pd.DataFrame(X_new_test, index=new_test[config['standardize_predictor_list']].index, columns=new_test[config['standardize_predictor_list']].columns)
    new_stand_train = pd.concat([new_stand_train, new_train[['season', config['target_variable'], 'weather_type', 'time_interval']]], axis = 1)
    new_stand_test = pd.concat([new_stand_test, new_test[['season','weather_type', 'time_interval']]], axis = 1)
    return new_stand_train, new_stand_test

# --------------------------------------------------- DEFINING MODELS ----------------------------------------


def train_XGB_regressor(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7, 10],
            'colsample_bytree': [0.3, 0.7]
    }
    grid_search = GridSearchCV(
        estimator=XGBRegressor(verbosity = 0),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=10,
    )
    grid_search.fit(X_train, y_train, verbose=0)
    return grid_search


def train_RF_regressor(X_train, y_train):
    param_grid = {
         'n_estimators': [50, 100, 200], 
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        estimator= RandomForestRegressor(),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=10,
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def train_MLP_regressor(X_train, y_train):
    param_grid = {
        'alpha': 10.0 ** -np.arange(1, 7),  # Regularization parameter
        'activation': ['relu'],  # Activation function
        'learning_rate_init': [0.1, 0.01, 0.001, 0.0001]  # Learning rate
        }

    grid_search = GridSearchCV(
        estimator=MLPRegressor(hidden_layer_sizes=(15,15), solver='adam', max_iter=10000),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=10,
    )

    grid_search.fit(X_train, y_train)
    return grid_search
# --------------------------------------------------- TRAINING AND PREDICTING MODELS ----------------------------------------

def train_predict_MLP_model(new_stand_train, new_stand_test):
    forecast_test = pd.DataFrame()
    for interval, weather_type in product(config['time_intervals'], config['weather_types']):
        X_train = new_stand_train[(new_stand_train['time_interval'] == interval) & (new_stand_train['weather_type'] == weather_type)][config['predictors']]
        y_train = new_stand_train[(new_stand_train['time_interval'] == interval) & (new_stand_train['weather_type'] == weather_type)][config['target_variable']]
        X_test = new_stand_test[(new_stand_test['time_interval'] == interval) & (new_stand_test['weather_type'] == weather_type)][config['predictors']]
        try:
            md = joblib.load(urlopen(f'https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/Fitted_Models/MLP_fitted_{interval}_{weather_type}.pkl'))
            #md = joblib.load(f'Fitted_Models\\MLP_fitted_{interval}_{weather_type}.pkl')
            predictions = md.predict(X_test)
        except:
            md = train_MLP_regressor(X_train, y_train)
            joblib.dump(md, f'Fitted_Models\\MLP_fitted_{interval}_{weather_type}.pkl') #save fitted model
            predictions = md.predict(X_test)
            
        print(f"Energy Predictions done for {interval, weather_type}")
        TestingData=pd.DataFrame(data=X_test.copy(), columns=X_test.columns)
        TestingData['PredictedTotalPower']=predictions
        forecast_test = pd.concat([forecast_test, TestingData])
    forecast_test = forecast_test.sort_index()
    return forecast_test


# Forecast data based on test
def train_predict_XGB_model(new_stand_train, new_stand_test):
    forecast_test = pd.DataFrame()
    for interval, weather_type in product(config['time_intervals'], config['weather_types']):
        X_train = new_stand_train[(new_stand_train['time_interval'] == interval) & (new_stand_train['weather_type'] == weather_type)][config['predictors']]
        y_train = new_stand_train[(new_stand_train['time_interval'] == interval) & (new_stand_train['weather_type'] == weather_type)][config['target_variable']]
        X_test = new_stand_test[(new_stand_test['time_interval'] == interval) & (new_stand_test['weather_type'] == weather_type)][config['predictors']]
        try:
            md = joblib.load(urlopen(f'https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/Fitted_Models/XGB_fitted_{interval}_{weather_type}.pkl'))
            #md = joblib.load(f'Fitted_Models\\XGB_fitted_{interval}_{weather_type}.pkl')
            predictions = md.predict(X_test)
        except:
            md = train_XGB_regressor(X_train, y_train)
            
            joblib.dump(md, f'Fitted_Models\\XGB_fitted_{interval}_{weather_type}.pkl') #save fitted model
            predictions = md.predict(X_test)
            
        print(f"Energy Predictions done for {interval, weather_type}")
        TestingData=pd.DataFrame(data=X_test.copy(), columns=X_test.columns)
        TestingData['PredictedTotalPower']=predictions
        forecast_test = pd.concat([forecast_test, TestingData])
    forecast_test = forecast_test.sort_index()
    return forecast_test

def train_predict_RF_model(new_stand_train, new_stand_test):
    forecast_test = pd.DataFrame()
    for interval, weather_type in product(config['time_intervals'], config['weather_types']):
        X_train = new_stand_train[(new_stand_train['time_interval'] == interval) & (new_stand_train['weather_type'] == weather_type)][config['predictors']]
        y_train = new_stand_train[(new_stand_train['time_interval'] == interval) & (new_stand_train['weather_type'] == weather_type)][config['target_variable']]
        X_test = new_stand_test[(new_stand_test['time_interval'] == interval) & (new_stand_test['weather_type'] == weather_type)][config['predictors']]
        try:
            md = joblib.load(urlopen(f'https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/Fitted_Models/RF_fitted_{interval}_{weather_type}.pkl'))
            #md = joblib.load(f'Fitted_Models\\RF_fitted_{interval}_{weather_type}.pkl')
            predictions = md.predict(X_test)
        except:
            md = train_RF_regressor(X_train, y_train)
            joblib.dump(md, f'Fitted_Models\\RF_fitted_{interval}_{weather_type}.pkl') #save fitted model
            predictions = md.predict(X_test)
            
        print(f"Energy Predictions done for {interval, weather_type}")
        TestingData=pd.DataFrame(data=X_test.copy(), columns=X_test.columns)
        TestingData['PredictedTotalPower']=predictions
        forecast_test = pd.concat([forecast_test, TestingData])
    forecast_test = forecast_test.sort_index()
    return forecast_test


# --------------------------------------------------- EVALUATING MODELS -----------------------------------------


# SMAPE
def smape(A, F):
    return 100/len(A) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F)))

# Evaluate model
def evaluate_model(forecast_test, test,new_test):
    y_test = test.loc[new_test.index, config['target_variable']]
    forecast_test['TotalPower']= y_test
    #set negative values at 0
    forecast_test['PredictedTotalPower'] = forecast_test['PredictedTotalPower'].clip(lower=0)

    mae = metrics.mean_absolute_error(forecast_test['TotalPower'], forecast_test['PredictedTotalPower'])
    mse = metrics.mean_squared_error(forecast_test['TotalPower'], forecast_test['PredictedTotalPower'])
    mape = metrics.mean_absolute_percentage_error(forecast_test['TotalPower'], forecast_test['PredictedTotalPower'])
    rmse = np.sqrt(mse) 
    r2 = metrics.r2_score(forecast_test['TotalPower'],forecast_test['PredictedTotalPower'])
    smape_ = smape(forecast_test['TotalPower'], forecast_test['PredictedTotalPower'])  
    print("Results of sklearn.metrics:")
    print("MAE:",round(mae,3))
    print("RMSE:", round(rmse,3))
    print(f"R-Squared: {round(r2,3) * 100} %")
    print(f"Scaled mean absolute percentage error: {round(smape_,3)} %")
    


def train_SHAP_values(df):
    """ This functions computes shap values for XGB and RF looping through each interval and weather type
        The output are the shap values in the form of a dataframe, to be used as a np array in subsequent summary plots"""
    XGB_shap_df = pd.DataFrame(columns = df.iloc[:, :-2].columns)
    RF_shap_df = pd.DataFrame(columns = df.iloc[:, :-2].columns)
    for interval, weather_type in product(config['time_intervals'], config['weather_types']):
        #subset data
        subset_data = df[(df['weather_type'] == weather_type) & (df['time_interval'] == interval)].iloc[:, :-2]
        #import models

        XGB_gr = joblib.load(urlopen(f'https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/Fitted_Models/XGB_fitted_{interval}_{weather_type}.pkl'))
        XGB_estimator = XGB_gr.best_estimator_
        RF_gr = joblib.load(urlopen(f'https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/Fitted_Models/RF_fitted_{interval}_{weather_type}.pkl'))
        RF_estimator = RF_gr.best_estimator_

        #compute shap values for current interval and weather type
        XGB_explainer = shap.TreeExplainer(XGB_estimator)
        RF_explainer = shap.TreeExplainer(RF_estimator)
        XGB_shap_values = XGB_explainer.shap_values(subset_data)
        RF_shap_values = RF_explainer.shap_values(subset_data)

        XGB_shap_subset = pd.DataFrame(XGB_shap_values, columns = subset_data.columns)
        RF_shap_subset = pd.DataFrame(RF_shap_values, columns = subset_data.columns)

        XGB_shap_df = pd.concat([XGB_shap_df, XGB_shap_subset])
        RF_shap_df = pd.concat([RF_shap_df, RF_shap_subset])

    return XGB_shap_df, RF_shap_df

def import_SHAP_values():
    MLP_shap_df = pd.read_csv("https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/Fitted_Models/MLP_Shap.csv", sep = '\t')
    XGB_shap_df = pd.read_csv("https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/Fitted_Models/XGB_Shap.csv", sep = '\t')
    RF_shap_df = pd.read_csv("https://raw.githubusercontent.com/Alex-Malainic/Solar-Energy/main/Fitted_Models/RF_Shap.csv", sep = '\t')
    return MLP_shap_df, XGB_shap_df, RF_shap_df

# Main order of functions
def main():
    df = load_data(config['file_path'])
    df = add_season(df)
    df = choose_interval(df)
    train, test = split_data(df)
    train = detect_time_interval(train)
    test = detect_time_interval(test)
    new_train_data = create_weather_type(train)
    new_test_data = classify_weather_type(new_train_data, test)
    new_stand_train, new_stand_test = standardize_data(new_train_data, new_test_data)
    forecasted_data_MLP = train_predict_MLP_model(new_stand_train, new_stand_test)
    evaluate_model(forecasted_data_MLP, test, new_test_data)
