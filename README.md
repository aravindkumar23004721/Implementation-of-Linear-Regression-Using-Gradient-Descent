# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights randomly. 
2. Compute predicted values. 
3. Compute gradient of loss function.
4. Update weights using gradient descent

## Program:
```python
/*
Program to implement the linear regression using gradient descent.
Developed by: ARAVIND KUMAR SS
RegisterNumber:212223110004
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta


data=pd.read_csv("/content/50_Startups.csv")
data.head()


X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)


theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![Screenshot 2024-03-05 182031](https://github.com/user-attachments/assets/22633521-64c7-4976-bc5d-b0d938d6f196)
![Screenshot 2024-03-05 182404](https://github.com/user-attachments/assets/b102fd45-1217-45b6-a0c2-5cdce0444219)
![Screenshot 2024-03-05 182438](https://github.com/user-attachments/assets/72903833-6fae-487f-8e84-768caf4bd342)
![Screenshot 2024-09-01 221209](https://github.com/user-attachments/assets/9663f9a0-74a1-4b7c-901b-6ea418af50e2)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
