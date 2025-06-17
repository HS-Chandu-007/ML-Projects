import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv("diabetes.csv")
df = pd.DataFrame(data)

#print(df.info())
#print(df.groupby('Outcome').mean())

x = df.drop(columns= 'Outcome', axis=1)
y = df['Outcome']
# print(x)
# print(y)

scaler = StandardScaler()
scaler.fit(x)
std_data = scaler.transform(x)

#print(std_data)

std_x = std_data
label_y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=50)
#print(x, x_train, x_test)

classifier = svm.SVC(kernel='linear')

classifier.fit(x_train, y_train)

x_train_prediction = classifier.predict(x_train)
training_data_prediction = accuracy_score(x_train_prediction, y_train)

#print(f"The Accuracy score of training data : {training_data_prediction*100}")

x_test_prediction = classifier.predict(x_test)
testing_data_prediction = accuracy_score(x_test_prediction, y_test)

#print(f"The Accuracy score of testing data : {testing_data_prediction*100}")

