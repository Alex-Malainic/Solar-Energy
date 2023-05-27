import streamlit as st 
import pandas as pd
import requests
import numpy as np
st.set_page_config(page_title = "Forecasting_Solar_Energy", page_icon = ':sun_small_cloud:', layout="wide")

st.title('Study - Forecasting Energy Generation Output for Solar Panels  :sun_small_cloud:')


Predictors = ['temperature_2m', 'relativehumidity_2m',
              'direct_radiation', 'diffuse_radiation',  'winddirection_10m', 'precipitation' ]


weather_df = pd.DataFrame(columns = Predictors )

#latitude and longitude of the panel
lat = -23.760363
long = 133.874719

#start and end date of the available solar power data
start = '2013-07-22'
end = '2023-05-17'

#historical weather for the panel
r = requests.get('https://archive-api.open-meteo.com/v1/archive', params={'latitude':lat, 'longitude': long, 'timezone': 'auto', 'start_date':start , 'end_date': end , 'hourly' : Predictors}).json() #timezone = auto so that it matches the local timezone

#create weather dataset
time = pd.to_datetime(np.array(r['hourly']['time']))
weather_df['time'] = time

for p in Predictors:
        weather_df[p] = np.array(r['hourly'][p])

weather_df['timestamp'] = pd.to_datetime(weather_df['time'])
weather_df.set_index('timestamp', inplace=True)

st.write(weather_df.head())
