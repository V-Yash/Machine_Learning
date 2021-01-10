import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt  
data_load = pd.read_excel("E:\code\PYTHON\Data Science\student_scores.xlsx")  
print("Successfully imported data into console" )
data_load.head(6)
data_load.plot(x='Hours', y='Scores', style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('The Hours Studied')    
plt.ylabel('The Percentage Score')    
plt.show()
X = data_load.iloc[:, :-1].values    
y = data_load.iloc[:, 1].values
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)   
from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
  
print("Training ... Completed !.") 
line = regressor.coef_*X+regressor.intercept_  
plt.scatter(X, y)  
plt.plot(X, line);  
plt.show()
print(X_test)   
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
df
hours = [[7]]  
own_pred = regressor.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0])) 