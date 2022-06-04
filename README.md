# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 1.Start the program 
2. Import the numpy.pandas and matplotlib 
3. Read the file which store the data  
4. declare and import train and test split, import StandardScaler and next ill do transform  train  and test fit
5. import logisitic rergession  to classifiying the train and test fit
6. using y predict to allocate and testing the x prediction
7. import confusion matrix to allocate cm=confusion_matrix(y_Test,y_Pred)
8. import metrices to allocate accuracy=metrics.accuracy_score(y_Test,y_Pred)
9. using recall of sensitivity and specificty recall_sensitivity = metrics.recall_score(y_Test,y_Pred,pos_label=1)
recall_specificity=metrics.recall_score(y_Test,y_Pred,pos_label=0)
10.  using matblotlib import ListedColormap finalized outputs 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S.Prakash Raaj
RegisterNumber:212220040120  
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datasets=pd.read_csv('/content/social_network.csv')
x=datasets.iloc[:,[2,3]].values
y=datasets.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_Train, x_Test,y_Train,y_Test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_x
StandardScaler()
x_Train=sc_x.fit_transform(x_Train)
x_Test=sc_x.transform(x_Test)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_Train,y_Train)
y_Pred=classifier.predict(x_Test)
y_Pred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_Test,y_Pred)
cm
from sklearn import metrics
accuracy=metrics.accuracy_score(y_Test,y_Pred)
accuracy
recall_sensitivity = metrics.recall_score(y_Test,y_Pred,pos_label=1)
recall_specificity=metrics.recall_score(y_Test,y_Pred,pos_label=0)
recall_sensitivity,recall_specificity
from matplotlib.colors import ListedColormap
x_Set,y_Set=x_Train,y_Train
x1,x2=np.meshgrid(np.arange(start=x_Set[:,0].min()-1,stop=x_Set[:,0].max()+1,step=0.01),np.arange(start=x_Set[:,1].min()-1,stop=x_Set[:,1].max(),step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','white')))

plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_Set)):
  plt.scatter(x_Set[y_Set==j,0],x_Set[y_Set==j,1],c=ListedColormap(('black','blue'))(i),label=j)
  plt.title('Logistic Regression(Training Set)')
  plt.xlabel('age')
  plt.ylabel('Estimated Salary')
  plt.legend()
  plt.show()

```

## Output:
![logistic regression using gradient descent](/ypredict.PNG)
![logistic regression using gradient descent](/confusionmatrix.PNG)
![logistic regression using gradient descent](/accuracy.PNG)
![logistic regression using gradient descent](/recall.PNG)
![logistic regression using gradient descent](/output1.PNG)
![logistic regression using gradient descent](/output2.PNG)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

