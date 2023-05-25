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
~df.duplicated()
df1=df[~df.duplicated()]
df1
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
df.skew()
df.kurtosis()
sns.boxplot(x="Ozone AQI Value",data=df)
sns.boxplot(x="PM2.5 AQI Value",data=df)
sns.countplot(x="AQI Value",data=df)
sns.distplot(df["AQI Value"])
sns.histplot(df["NO2 AQI Value"])
sns.displot(df["CO AQI Value"])
sns.scatterplot(x=df['AQI Value'],y=df['NO2 AQI Value'])
import matplotlib.pyplot as plt
states=df.loc[:,["AQI Category","AQI Value"]]
states=states.groupby(by=["AQI Category"]).sum().sort_values(by="AQI Value")
plt.figure(figsize=(17,7))
sns.barplot(x=states.index,y="AQI Value",data=states)
plt.xlabel=("AQI Category")
plt.ylabel=("AQI Value")
plt.show()
import matplotlib.pyplot as plt
states=df.loc[:,["CO AQI Category","CO AQI Value"]]
states=states.groupby(by=["CO AQI Category"]).sum().sort_values(by="CO AQI Value")
plt.figure(figsize=(17,7))
sns.barplot(x=states.index,y="CO AQI Value",data=states)
plt.xlabel=("CO AQI Category")
plt.ylabel=("CO AQI Value")
plt.show()
df.corr()
sns.heatmap(df.corr(),annot=True)
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
df['AQI']=le.fit_transform(df['AQI Value'])
df
le=LabelEncoder()
df['CO AQI']=le.fit_transform(df['CO AQI Value'])
df
le=LabelEncoder()
df['PM2.5 AQI']=le.fit_transform(df['PM2.5 AQI Value'])
df
AQI=['Good','Moderate','Unhealthy','Unhealthy for Sensitive Groups','Very Unhealthy','Hazardous']
enc=OrdinalEncoder(categories=[AQI])
enc.fit_transform(df[['AQI Category']])
df['AQI CATEGORY']=enc.fit_transform(df[['AQI Category']])
df
AQI_New=['Good','Moderate','Unhealthy','Unhealthy for Sensitive Groups','Very Unhealthy','Hazardous']
enc=OrdinalEncoder(categories=[AQI_New])
enc.fit_transform(df[['Ozone AQI Category']])
df['OZONE AQI CATEGORY']=enc.fit_transform(df[['Ozone AQI Category']])
df
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()
from sklearn.preprocessing import OneHotEncoder
df1 = pd.read_csv("Air Quality.csv")
ohe=OneHotEncoder(sparse=False)
enc=pd.DataFrame(ohe.fit_transform(df1[['CO AQI Category']]))
df1=pd.concat([df1,enc],axis=1)
df1
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
sm.qqplot(df1['AQI Value'],fit=True,line='45')
plt.show()
sm.qqplot(df1['PM2.5 AQI Value'],fit=True,line='45')
plt.show()
import numpy as np
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df1['NO2 AQI Value']=pd.DataFrame(transformer.fit_transform(df1[['NO2 AQI Value']]))
sm.qqplot(df1['NO2 AQI Value'],line='45')
plt.show()
transformer=PowerTransformer("yeo-johnson")
df1['CO AQI Value']=pd.DataFrame(transformer.fit_transform(df1[['CO AQI Value']]))
sm.qqplot(df1['CO AQI Value'],line='45')
plt.show()
qt=QuantileTransformer(output_distribution='normal')
df1['AQI Value']=pd.DataFrame(qt.fit_transform(df1[['AQI Value']]))
sm.qqplot(df1['AQI Value'],line='45')
plt.show()
df1.drop([0,1,2],axis=1, inplace=True)
df1
sns.barplot(x="CO AQI Category",y="CO AQI Value",data=df1)
plt.xticks(rotation = 90)
plt.show()
sns.barplot(x="PM2.5 AQI Category",y="PM2.5 AQI Value",data=df1)
plt.xticks(rotation = 90)
plt.show()
sns.lineplot(x="CO AQI Value",y="NO2 AQI Category",data=df1,hue="AQI Category",style="AQI Category")
sns.scatterplot(x="AQI Value",y="NO2 AQI Value",hue="AQI Category",data=df1)
sns.histplot(data=df1, x="PM2.5 AQI Value", hue="PM2.5 AQI Category", element="step", stat="density")
sns.relplot(data=df1,x=df1["AQI Category"],y=df1["AQI Value"],hue="AQI Category")
sns.relplot(data=df1,x=df1["CO AQI Category"],y=df1["CO AQI Value"],hue="CO AQI Category")
