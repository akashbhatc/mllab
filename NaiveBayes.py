import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB as Classifier
from seaborn import load_dataset
def evaluate_naive_bayes(dataset,target_col,K_values):
    for K in K_values:
        x_train,x_test,y_train,y_test=train_test_split(dataset.drop(columns=[target_col]),dataset[target_col],test_size=(1-K/10),random_state=42)
        Clf=Classifier()
        Clf.fit(x_train.values,y_train.values)
        y_pred=Clf.predict(x_test.values)
        accuracy=accuracy_score(y_test,y_pred)
        print(f"Split {K*10}-{100-K*10} Accuracy:{accuracy:.2f}")
titanic_data=pd.read_csv(r"C:\Users\Akash\Downloads\titanic.csv")
titanic_data=titanic_data.drop(columns=['PassengerId','Name','Ticket','Cabin'])
titanic_data['Age'].fillna(titanic_data['Age'].median(),inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)
titanic_data=pd.get_dummies(titanic_data,columns=['Sex','Embarked'])
print("\nTitanic dataset Naive Bayes Classifier Results: ")
evaluate_naive_bayes(titanic_data,'Survived',[3,5,7,9])
