#importing the libarires we need to build this "Rock vs Mine" Prediction system


import numpy as np # numpy for numerical analysis
import pandas as pd # pandas for data manipulation and data preprocessing
import sklearn # no need to import sklearn cause we are importing specific needed functions from it down below. Just to avoid any error we are importing it here
from sklearn.model_selection import train_test_split # to split our data into training data and testing data
from sklearn.linear_model import LogisticRegression # this is our ML model
from sklearn.metrics import accuracy_score # we use this test accuracy % of the ML model


#here we are reading our csv file and we dont have header files in dataset so we gave it none
data = pd.read_csv("sonar data.csv", header=None) 

#we are converting our csv file into pandas dataframe
new_data = pd.DataFrame(data)

#this gives us the statistical measures of our data set
print(new_data.describe())

#here we are counting how many rocks vs mines are there 
new_data[60].value_counts()

#now we are taking mean of the last coloum which is labeled as R and M so we get the relation between them 
new_data.groupby(60).mean()

#here removing the last coloum and storing it in a new variable y
x = new_data.drop(columns=60, axis=1)
y = new_data[60]

#now we are spliting our data in testing and training
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1, random_state= 1, stratify=y)

# the ML is stored in model variable
model = LogisticRegression()

#we are just simply feeding the data into our "LogisticRegression()" model so it can learn from it
model.fit(x_train, y_train)

# accuracy on training data
x_train_prediction = model.predict(x_train)
y_train_accuracy = accuracy_score(x_train_prediction, y_train) 

print(f"Accuracy on training data : {y_train_accuracy*100}")

# accuracy on test data

x_test_prediction = model.predict(x_test)
y_test_accuracy = accuracy_score(x_test_prediction, y_test)
#print(f"Accuracy on testing data : {y_test_accuracy*100}")

input_data = (0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115
)

new_input_data = np.asarray(input_data)

reshaped_data = new_input_data.reshape(1, -1)

predict = model.predict(reshaped_data)

#print(predict)

if(predict == 'R'):
    print(f" '{predict}' It is a Rock.")
else:
    print(f" '{predict}' It is a Mine.")