# EX-NO-10-DATA SCIENCE PROCESS ON COMPLEX DATASET
# AIM:
  To Perform Data Science Process on a complex dataset and save the data to a file.
# ALGORITHM:
## STEP-1:
  Read the given Data.
## STEP-2:
  Clean the Data Set using Data Cleaning Process.
## STEP-3:
Apply Feature Generation/Feature Selection Techniques on the data set.
## STEP-4:
Apply EDA /Data visualization techniques to all the features of the dataset.
# CODE:
## Data Cleaning Process:
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("Air Quality.csv")

df.head(10)

df.info()

df.describe()

df.isnull().sum()
## Handling Outliers:
q1=df['PM2.5 AQI Value'].quantile(0.25)

q3=df['PM2.5 AQI Value'].quantile(0.75)

IQR=q3-q1

print("First quantile:",q1," Third quantile:",q3," IQR: ",IQR,"\n")

lower=q1-1.5*IQR

upper=q3+1.5*IQR

outliers=df[(df['PM2.5 AQI Value']>=lower)&(df['PM2.5 AQI Value']<=upper)]

from scipy.stats import zscore

z=outliers[(zscore(outliers['PM2.5 AQI Value'])<3)]

print("Cleaned Data: \n")

print(z)
## EDA Techniques:
df.skew()

df.kurtosis()

sns.boxplot(x="Ozone AQI Value",data=df)

sns.countplot(x="AQI Value",data=df)

sns.distplot(df["AQI Value"])

sns.histplot(df["NO2 AQI Value"])

sns.displot(df["CO AQI Value"])

sns.scatterplot(x=df['AQI Value'],y=df['NO2 AQI Value'])

states=df.loc[:,["AQI Category","AQI Value"]]

states=states.groupby(by=["AQI Category"]).sum().sort_values(by="AQI Value")

plt.figure(figsize=(17,7))

sns.barplot(x=states.index,y="AQI Value",data=states)

plt.xlabel=("AQI Category")

plt.ylabel=("AQI Value")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)
## Feature Generation:
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

le=LabelEncoder()

df['AQI']=le.fit_transform(df['AQI Value'])

df

AQI=['Good','Moderate','Unhealthy','Unhealthy for Sensitive Groups','Very Unhealthy','Hazardous']

enc=OrdinalEncoder(categories=[AQI])

enc.fit_transform(df[['AQI Category']])

df['AQI CATEGORY']=enc.fit_transform(df[['AQI Category']])

df

ohe=OneHotEncoder(sparse=False)

enc=pd.DataFrame(ohe.fit_transform(df1[['CO AQI Category']]))

df1=pd.concat([df1,enc],axis=1)

df1
## Feature Transformation:
import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import PowerTransformer

sm.qqplot(df1['AQI Value'],fit=True,line='45')

plt.show()

transformer=PowerTransformer("yeo-johnson")

df1['NO2 AQI Value']=pd.DataFrame(transformer.fit_transform(df1[['NO2 AQI Value']]))

sm.qqplot(df1['NO2 AQI Value'],line='45')

plt.show()

qt=QuantileTransformer(output_distribution='normal')

df1['AQI Value']=pd.DataFrame(qt.fit_transform(df1[['AQI Value']]))

sm.qqplot(df1['AQI Value'],line='45')

plt.show()
## Data Visualization:
sns.barplot(x="CO AQI Category",y="CO AQI Value",data=df1)

plt.xticks(rotation = 90)

plt.show()

sns.lineplot(x="CO AQI Value",y="NO2 AQI Category",data=df1,hue="AQI Category",style="AQI Category")

sns.scatterplot(x="AQI Value",y="NO2 AQI Value",hue="AQI Category",data=df1)

sns.relplot(data=df1,x=df1["CO AQI Category"],y=df1["CO AQI Value"],hue="CO AQI Category")

sns.histplot(data=df1, x="PM2.5 AQI Value", hue="PM2.5 AQI Category",element="step", stat="density")
# OUTPUT:
## Data Cleaning Process:
![Screenshot (92)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/d4b5d05c-51da-48a1-8885-9902d0a765f9)

![Screenshot (94)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/61faf55a-f708-40cf-b3e1-cab4400bc571)

![Screenshot (93)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/6e7de91b-e503-4a2e-8a6b-d4fe11ea297f)

![Screenshot (95)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/a6b567a2-c65b-472c-94bf-69018bca0bd9)
## Handling Outliers:
![Screenshot (96)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/666748f7-6edc-42b1-9f10-dd098305fee5)
## EDA Techniques:
![Screenshot (98)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/575cb18a-a7b4-40eb-b001-13534095d44b)

![Screenshot (99)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/9dffecbe-1879-4b3e-93e5-df5b093cafac)

![Screenshot (100)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/be75c186-165c-45f8-a855-b9ecf188e26a)

![Screenshot (101)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/02bd5cd9-721a-4f58-a832-9591c16b7bfa)

![Screenshot (102)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/c815fd72-4735-4bd0-8577-847b10fb8395)

![Screenshot (103)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/31b3aa7f-7bca-4d91-82dd-d44a6f69e59a)

![Screenshot (104)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/bf89673f-627e-4d10-9135-b0f62b1f21f8)

![Screenshot (105)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/bf7d5ed4-0aa2-4d5d-b352-d8f5e87d7a9f)

![Screenshot (106)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/f393dfeb-fa5a-4f7e-bede-de28745f0081)

![Screenshot (107)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/ccd0d433-719c-4734-97f0-5095add3c36d)

![Screenshot (108)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/28929bdd-1c0e-4544-9531-fe8a96e61382)
## Feature Generation:
![Screenshot (109)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/446fe712-40ef-4598-aec9-8875dadce6ac)

![Screenshot (110)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/728ac5be-2e88-448b-9e60-01139d1bca9e)

![Screenshot (111)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/94a947c2-4faf-4726-ad9e-2133f2e820ee)
## Feature Transformation:
![Screenshot (112)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/936cbd9f-e21c-4aff-b93f-2369a127dc01)

![Screenshot (113)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/36ee8375-9010-484a-9daa-08894694cea4)

![Screenshot (114)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/7ea43c82-44ed-4167-9e12-8a997a28ee4d)
## Data Visualization:
![Screenshot (115)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/4a871319-ba03-4356-8150-c8993f16432c)

![Screenshot (116)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/adff18e9-329a-4c26-b94f-1725202f6f6d)

![Screenshot (117)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/fd80248b-3b4a-4c52-b4b9-b2005b2f03a2)

![Screenshot (119)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/d238e6f9-15e3-4626-abce-0a6ff781d323)

![Screenshot (118)](https://github.com/MaheshS03/EX-NO-10-DS/assets/128498431/b55ba915-aac4-462f-a4ab-87db48ba2c94)
# RESULT:
Thus the Data Science Process on Complex Dataset were performed and output was verified successfully.
