import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

df=pd.read_csv("dataSetFull.csv")
df=df.dropna(axis=0)
df=df.drop(["price_tnd","location","city","age"],axis=1)
x=df.iloc[:,2:].values
y=df.iloc[:,1].values
#7abitch ti5dim donc bich na7iha 
#cl = ColumnTransformer([("governorate", OneHotEncoder(), [24])], remainder = 'passthrough')
#x = cl.fit_transform(x)
#donc na7it parti ta3 governorate

x=x[:,1:]
print(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=1)
fx=LinearRegression()
fx.fit(x_train,y_train)
y_pred=fx.predict(x_test)
print(r2_score(y_test,y_pred))
print( fx.predict(np.array([1899,20,5,7,1,35.8569,10.5972,113.681036216882,0,0,0,0,0,1,0,0,1,1,1]).reshape(1,-1)))

