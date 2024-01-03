
# **Capstone Project**

This project is about 'KNN' Regression and Classification.
In this, I have completed two projects.
- Iphone project
* Houseprice project
---
##  **1 . Iphone project**
### _Problem statement_

Iphone Purchases are getting increased day by day and many stores wants to predict whether a customer will purchase an Iphone from thier store given their gender, age and salary.

### _Method_
I used `KNN Classifire` to predict whether a customer will purchase an iphone or not.

### This code imports `KNN Classifire`

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier
model_clf = KNeighborsClassifier(n_neighbors=9)
model_clf_train  = model_clf.fit(X_train,y_train)
y_pred = model_clf_train.predict(X_test)
model_clf_train.score(X_train,y_train) 
``` 


## [KNN Classifier Notebook]('https://github.com/vaibhavkatkar3001/Capstone-Project/blob/main/iphone_prj5.ipynb')

---

### _Business impact_

So I created a model using KNN which can predict wheather person can purchase Iphone or not. And the 86 % of accuracy is tell us that it is a pretty fair fitt the model.


---



##  **2 . Houseprice project**
### _Problem statement_

To Predict the Price of Bangalore House.

---
### _Method_
I used `KNN Regressor` to predict price of Bangalore house

### This code imports `Regressor`

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,y_train)
y_pred = model_fit.predict(X_test)
results = r2_score(y_test,y_pred)
print(results) 
``` 
---

## [KNN Regressor Notebook]('https://github.com/vaibhavkatkar3001/Capstone-Project/blob/main/houseprice_prj6.ipynb')

---
### _Business impact_
So, we created a model using KNN which can Predict the Price of Bangalore House. And the 96 % of accuracy is tell us that it is a pretty fair fit the model.

---
