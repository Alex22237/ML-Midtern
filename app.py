import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/train.csv")
df.info()
df.head()
sns.pairplot(df[['ShoppingMall', 'Transported']], dropna=True)
#Name and shoppingmall
df.drop(['Name', 'ShoppingMall'], axis=1, inplace=True)

#補東西
#Homeplanet
df['HomePlanet'].fillna(df['HomePlanet'].value_counts().idxmax(), inplace=True)
#CryoSleep
df['CryoSleep'].fillna(df['CryoSleep'].value_counts().idxmax(), inplace=True)
#Destination
df['Destination'].fillna(df['Destination'].value_counts().idxmax(), inplace=True)
#VIP
df['VIP'].fillna(df['VIP'].value_counts().idxmax(), inplace=True)
#RoomService
df['RoomService'] = df.groupby('Transported')['RoomService'].apply(lambda x: x.fillna(x.median()))
#FoodCourt
df['FoodCourt'] = df.groupby('Transported')['FoodCourt'].apply(lambda x: x.fillna(x.median()))
#Spa
df['Spa'] = df.groupby('Transported')['Spa'].apply(lambda x: x.fillna(x.median()))
#VRDeck
df['VRDeck'] = df.groupby('Transported')['VRDeck'].apply(lambda x: x.fillna(x.median()))
#Age
print(df['Age'].mean())
df['Age'].fillna(df['Age'].mean())
df['Age'] = df.groupby('Transported')['Age'].apply(lambda x: x.fillna(x.mean()))
#Cabin
from sklearn import preprocessing
df[['cabinA', 'cabinB', 'cabinC']] = df['Cabin'].str.split('/', expand=True)
label_encoder = preprocessing.LabelEncoder()
df['encoded_CabinA'] = label_encoder.fit_transform(df['cabinA'])
df['encoded_CabinB'] = label_encoder.fit_transform(df['cabinB'])
df.drop(['cabinA'], axis=1,inplace=True)
df.drop(['cabinB'], axis=1,inplace=True)
df.drop(['cabinC'], axis=1,inplace=True)
df.drop(['Cabin'], axis=1,inplace=True)
sns.pairplot(df[['encoded_CabinA', 'Transported']], dropna=True)
sns.pairplot(df[['encoded_CabinB', 'Transported']], dropna=True)
df.drop(['encoded_CabinA'], axis=1,inplace=True)
df.drop(['encoded_CabinB'], axis=1,inplace=True)

#object轉數字
df = pd.get_dummies(data=df, columns=['CryoSleep', 'VIP'])
df.drop(['CryoSleep_False', 'VIP_False'], axis=1, inplace=True)

#HomePlanet & Destination
df['HomePlanet'].unique()
df['Destination'].unique()
df['HomePlanet'].value_counts()
df['Destination'].value_counts()
df = pd.get_dummies(data=df, columns=['HomePlanet', 'Destination'])

#確認資料與結果相關性
df.corr()['Transported'].sort_values(ascending=False)
#丟掉絕對值<0.05的資料
df.drop(['VIP_True', 'Destination_PSO J318.5-22', 'HomePlanet_Mars', 'FoodCourt'], axis=1, inplace=True)


x = df.drop(['Transported'], axis=1)
y= df['Transported']
df.drop(['Transported'], axis=1, inplace=True)

#choose model & train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3, random_state=87)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=150) 
lr.fit(x_train,y_train)
predictions= lr.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
accuracy_score(y_test,predictions)
recall_score(y_test,predictions)
precision_score(y_test,predictions)
confusion_matrix(y_test, predictions)
pd.DataFrame(confusion_matrix(y_test, predictions), columns=['not Transported', 'Transported' ])

import joblib
joblib.dump(lr, 'spaceship-20230414.pkl', compress=3)



