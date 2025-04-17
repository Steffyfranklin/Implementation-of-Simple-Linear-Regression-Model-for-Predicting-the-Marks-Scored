# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Steffy Aavlin Raj F S 
RegisterNumber:212224040330
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_test

plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test, Y_test, color="purple")
plt.plot(X_test, regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test, Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test, Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE",rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/e5b7d5c5-7898-489a-9bcf-2e17a9bf2daa)

![image](https://github.com/user-attachments/assets/3abd758c-ad16-4a40-9519-a4c028371369)

![image](https://github.com/user-attachments/assets/96dc6192-41db-4ace-ad38-30f94b421a0f)

![image](https://github.com/user-attachments/assets/be028bce-dd47-4ad5-b7f7-d6b842b0beff)
![image](https://github.com/user-attachments/assets/75c4741c-d1d1-4c57-acab-b427d7075151)
![image](https://github.com/user-attachments/assets/96f1ce63-9177-434f-84e7-f446bdcef70d)
![image](https://github.com/user-attachments/assets/3895014a-5879-48d9-9abb-2060547dcadf)
![image](https://github.com/user-attachments/assets/a4ef9940-30ed-4395-95a3-1e8890b93cf0)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
