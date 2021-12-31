# General and EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ML Algorithms
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

# Metric
from sklearn.metrics import mean_absolute_error
from sklearn import metrics

# Streamlit
import streamlit as st

st.write("""
# House Price Prediction App
This app predicts the **Dhaka House Price**!
""")
st.write('---')

# Read dataset
df = pd.read_csv('housing_dataset_bd.csv')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

# Drop null values
df.dropna(axis=0, inplace=True)

# Reset the index after dropping the null values
df.reset_index(drop=True, inplace=True)

st.header('Geographic View of Model Training Data')
st.write("""
This model is trained on House prices in Dhaka
""")
st.map(df)
st.write('---')

# Select features and set target
X = df.drop(['Price', 'Location', 'Type', 'Region', 'Sub_region'], axis=1)
y = df['Price']

# Scale data in a scale of 0-1
scaler = MinMaxScaler()
scaler.fit_transform(X)


# Side bar slider for user input
def user_input_features():
    No_Beds = st.sidebar.slider(
        'No. Beds', int(X.No_Beds.min()), int(X.No_Beds.max()), int(X.No_Beds.mean()))
    No_Baths = st.sidebar.slider(
        'No. Baths', int(X.No_Baths.min()), int(X.No_Baths.max()), int(X.No_Baths.mean()))
    Area = st.sidebar.slider('Area (Sq.ft.)', float(X.Area.min()), float(
        X.Area.max()), float(X.Area.mean()))
    latitude = st.sidebar.slider(
        'Latitude', float(X.latitude.min()), float(X.latitude.max()), float(X.latitude.mean()))
    longitude = st.sidebar.slider(
        'Longitude', float(X.longitude.min()), float(X.longitude.max()), float(X.longitude.mean()))

    data = {'No. Beds': No_Beds,
            'No. Baths': No_Baths,
            'Area': Area,
            'Latitude': latitude,
            'Longitude': longitude}
    features = pd.DataFrame(data, index=[0])
    return features


df_user_ip = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df_user_ip)
st.write('---')


total_pred = 0
list_pred = []

# Model
# KNN Regressor
neigh = KNeighborsRegressor(
    n_neighbors=10, weights='uniform', algorithm='auto', p=1)
neigh.fit(X, y)
y_pred_1 = neigh.predict(df_user_ip)
list_pred.append(y_pred_1)
total_pred += y_pred_1

# DecisionTreeRegressor
regr_1 = DecisionTreeRegressor(max_depth=10)
regr_1.fit(X, y)
y_pred_2 = regr_1.predict(df_user_ip)
list_pred.append(y_pred_2)
total_pred += y_pred_2

# AdaBoostRegressor
regr_2 = AdaBoostRegressor(
    n_estimators=50, learning_rate=0.3, loss='exponential')
regr_2.fit(X, y)
y_pred_3 = regr_2.predict(df_user_ip)
list_pred.append(y_pred_3)
total_pred += y_pred_3

# RandomForestRegressor
regr_3 = RandomForestRegressor()
regr_3.fit(X, y)
y_pred_4 = regr_3.predict(df_user_ip)
list_pred.append(y_pred_4)
total_pred += y_pred_4

# Mean price prediction
mean_pred = total_pred / 4

st.header('Prediction of House Price (BDT)')
st.write(mean_pred)
st.write('---')

# Model comparison
st.header('Model Prediction Comparison')
st.write('This model uses 4 models- KNeighborsRegressor, DecisionTreeRegressor, AdaBoostRegressor, and RandomForestRegressor')
df_pred = pd.DataFrame(
    list_pred, index=['KNN', 'DecisionTree', 'AdaBoost', 'RandomForest'])
st.bar_chart(df_pred)
