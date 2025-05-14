# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.

   

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: AHALYA S
RegisterNumber: 212223230006
*/
```
```
import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

![image](https://github.com/user-attachments/assets/dc3c7473-33dc-4ee7-b248-348d65059d3e)

![image](https://github.com/user-attachments/assets/5f955361-5f44-4768-a896-5c241576ae71)

![image](https://github.com/user-attachments/assets/4efaee77-11f3-4349-8ee2-281132b24dbc)

![image](https://github.com/user-attachments/assets/ba6ae556-ba01-4f76-b218-0a5e82737437)

![image](https://github.com/user-attachments/assets/5744b072-b342-4e1b-9b94-9da5a68af4c5)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
