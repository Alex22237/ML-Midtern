import joblib
model_pretrained = joblib.load('spaceship-20230414.pkl')
import pandas as pd

df_test = pd.read_csv("Data/test.csv")
df_test.drop(['Name', 'ShoppingMall'], axis=1, inplace=True)
df_test.info()

#補東西
#Homeplanet
df_test['HomePlanet'].fillna(df_test ['HomePlanet'].value_counts().idxmax(), inplace=True)
#CryoSleep
df_test['CryoSleep'].fillna(df_test ['CryoSleep'].value_counts().idxmax(), inplace=True)
#Destination
df_test['Destination'].fillna(df_test ['Destination'].value_counts().idxmax(), inplace=True)
#VIP
df_test['VIP'].fillna(df_test ['VIP'].value_counts().idxmax(), inplace=True)
#RoomService
#因為不能使用有沒有Trasported這兩個群組的中位數進行填補，所以用此資料的眾數填補
df_test['RoomService'].fillna(df_test['RoomService'].mode()[0],inplace=True)
#FoodCourt
df_test['FoodCourt'].fillna(df_test['FoodCourt'].mode()[0],inplace=True)
#Spa
df_test['Spa'].fillna(df_test['Spa'].mode()[0],inplace=True)
#VRDeck
df_test['VRDeck'].fillna(df_test['VRDeck'].mode()[0],inplace=True)
#Age
df_test['Age'].fillna(df_test['Age'].mode()[0],inplace=True)
#Cabin
from sklearn import preprocessing
df_test [['cabinA', 'cabinB', 'cabinC']] = df_test ['Cabin'].str.split('/', expand=True)
label_encoder = preprocessing.LabelEncoder()
df_test ['encoded_CabinA'] = label_encoder.fit_transform(df_test ['cabinA'])
df_test ['encoded_CabinB'] = label_encoder.fit_transform(df_test ['cabinB'])
df_test.drop(['cabinA'], axis=1,inplace=True)
df_test.drop(['cabinB'], axis=1,inplace=True)
df_test.drop(['cabinC'], axis=1,inplace=True)
df_test.drop(['Cabin'], axis=1,inplace=True)
df_test.drop(['encoded_CabinA'], axis=1,inplace=True)
df_test.drop(['encoded_CabinB'], axis=1,inplace=True)

#object轉數字
df_test = pd.get_dummies(data=df_test , columns=['CryoSleep', 'VIP'])
df_test.drop(['CryoSleep_False', 'VIP_False'], axis=1, inplace=True)

#HomePlanet & Destination
df_test = pd.get_dummies(data=df_test, columns=['HomePlanet', 'Destination'])

#確認資料與結果相關性
#df_test.corr()['Transported_True'].sort_values(ascending=False)
#df_test['Transported'] = label_encoder.fit_transform(df_test['Transported_True'])
#丟掉絕對值<0.05的資料
df_test.drop(['VIP_True', 'Destination_PSO J318.5-22', 'HomePlanet_Mars', 'FoodCourt'], axis=1, inplace=True)

predictions2 = model_pretrained.predict(df_test)
predictions2

forSubmissionDF = pd.DataFrame(columns=['PassengerID', 'Transported'])
forSubmissionDF
forSubmissionDF['PassengerID'] = df_test['PassengerId']
forSubmissionDF['Transported'] = predictions2

forSubmissionDF.to_csv('For_Sudmission_20230414.csv', index=False)