#k fold cross validation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import numpy as np
digits = load_digits()
x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size = 0.2)

lr = LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_test,y_test)



 
