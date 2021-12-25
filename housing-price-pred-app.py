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

# Metric
from sklearn.metrics import mean_absolute_error

# Data
df = pd.read_csv("../input/california-housing-prices/housing.csv")
df

# EDA
df.info()
df.describe()
df[df['total_bedrooms'].isnull()==True]
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df

df.info()
df['median_house_value'].hist()

# Binning median_house_value  
df['housing_category'] = pd.cut(df['median_house_value'], bins=4, labels=['Low', 'Mid', 'High', 'Lavish'])
df

plt.figure(figsize=(25,9))
sns.scatterplot(x='latitude', y='longitude', hue='housing_category', data=df)

# Data fremes for each category og houses
dflow = df.loc[df['housing_category']=='Low']
dfmid = df.loc[df['housing_category']=='Mid']
dfhigh = df.loc[df['housing_category']=='High']
dflavish = df.loc[df['housing_category']=='Lavish']

plt.figure(figsize=(20,10))
plt.plot(dflow['latitude'], dflow['longitude'], 'o', label='Low')
plt.plot(dfmid['latitude'], dfmid['longitude'], 'o', label='Mid')
plt.plot(dfhigh['latitude'], dfhigh['longitude'], 'o', label='High')
plt.plot(dflavish['latitude'], dflavish['longitude'], 'ko', label='Lavish')
plt.legend()

plt.figure(figsize=(15,5))
plt.subplot(121)
sns.scatterplot(x='latitude', y='median_house_value', hue='housing_category', data=df)
plt.subplot(122)
sns.scatterplot(x='longitude', y='median_house_value', hue='housing_category', data=df)

plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.scatter3D(dflow['latitude'], dflow['longitude'], dflow['median_house_value'])
ax.scatter3D(dfmid['latitude'], dfmid['longitude'], dfmid['median_house_value'])
ax.scatter3D(dfhigh['latitude'], dfhigh['longitude'], dfhigh['median_house_value'])
ax.scatter3D(dflavish['latitude'], dflavish['longitude'], dflavish['median_house_value'])
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Median_house_value')

sample1 = df.loc[(df['latitude']>=34) & (df['latitude']<=34.02)]
sample1

sample1low = sample1.loc[sample1['housing_category']=='Low']
sample1mid = sample1.loc[sample1['housing_category']=='Mid']
sample1high = sample1.loc[sample1['housing_category']=='High']
sample1lavish = sample1.loc[sample1['housing_category']=='Lavish']

plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.scatter3D(sample1low['latitude'], sample1low['longitude'], sample1low['median_house_value'])
ax.scatter3D(sample1mid['latitude'], sample1mid['longitude'], sample1mid['median_house_value'])
ax.scatter3D(sample1high['latitude'], sample1high['longitude'], sample1high['median_house_value'])
ax.scatter3D(sample1lavish['latitude'], sample1lavish['longitude'], sample1lavish['median_house_value'])
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Median_house_value')

sns.heatmap(df.corr(), annot=True, cmap='Blues')

sns.heatmap(df.corr(method='spearman'), annot=True, cmap='Blues')

sns.heatmap(df.corr(method='kendall'), annot=True, cmap='Blues')

# KNN Regressor
X = df.drop(['median_house_value', 'housing_category', 'ocean_proximity'], axis=1)
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = MinMaxScaler()
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)

# Price predictions based on 10 nearest house prices and its other features.

neigh = KNeighborsRegressor(n_neighbors=10, weights='uniform', algorithm='auto', p=1)
neigh.fit(X_train, y_train)

y_train_pred = neigh.predict(X_train)
mean_absolute_error(y_train, y_train_pred)

y_pred = neigh.predict(X_test)
mean_absolute_error(y_test, y_pred)

# DecisionTreeRegressor
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(X_train, y_train)

# Predict
y_1 = regr_1.predict(X_test)
mean_absolute_error(y_test, y_1)

# AdaBoostRegressor
regr_2 = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4), n_estimators=800, random_state=10
)
regr_2.fit(X_train, y_train)

#predict
y_2 = regr_2.predict(X_test)
mean_absolute_error(y_test, y_2)
