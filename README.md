# Solar Energy Forecast

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A web application for PV energy forecast using meteorological data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Acess](#access)
- [Usage](#usage)
- [Credits](#credits)

## Introduction

This web application provides a PV (Photovoltaic) energy forecast based on meteorological data. It leverages machine learning algorithms to predict solar energy production for a given location and time frame. The forecast can help users optimize energy consumption and make informed decisions.

## Features

- Accurate PV energy forecast based on meteorological data
- User-friendly interface with intuitive controls
- Customizable parameters for EDA
- Historical data analysis for performance evaluation
- Real-time hourly forecast for the next 3 days using a weather API

## Access

The web app can be accessed at the following link: https://pv-energy-forecast.streamlit.app/

## Usage

**Home** - this page features a PV panel location explorer and an interactive dataset explorer.  
**EDA** - Exploratory Data Analysis with an interactive way to choose the data granularity and daytime (07:00 AM - 18:00 PM) or full time dataset.  
**ML Model Estimation** - this page features metrics for the three ML models (MLP, XGBoost and Random Forest), deviation plots and feature value importance using SHAP VALUES  
**Forecast** - in this page the MLP model forecast can be viewed for the next 3 days; the forecast is retrieved from the weather API.  

## Credits

Credits to https://dkasolarcentre.com.au/ for providing the PV energy historical dataset.


