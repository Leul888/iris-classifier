from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle

iris = load_iris()
X = iris.data
y= iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=.2)
my_model = RandomForestClassifier()
my_model.fit(X_train,y_train)
pred = my_model.predict(X_test)
accuracy = accuracy_score(pred,y_test)
# print(f'Accuracy:{accuracy:.2f}')

with open('Iris_model.pkl','wb') as f:
    pickle.dump(my_model,f)
print('Model saved as Iris_model.pkl')