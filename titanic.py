#Titanic
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

training_data_path = '/Users/raviachan/Documents/Programming/Kaggle/Titanic/titanic/train.csv'
training_data = pd.read_csv(training_data_path)

test_data_path = '/Users/raviachan/Documents/Programming/Kaggle/Titanic/titanic/test.csv'
test_data = pd.read_csv(test_data_path)

pd.set_option('display.max_columns', None)

#Training Data, getting X
training_data['CabinClass'] = training_data['Cabin'].astype(str).str[0]
training_data = pd.concat([training_data, pd.get_dummies(training_data['Sex'], prefix='Sex')], axis =1)
training_data = pd.concat([training_data, pd.get_dummies(training_data['CabinClass'], prefix='Cabin')], axis=1)
training_data.drop(['Cabin_T', 'Cabin_n', 'Sex', 'Name', 'Cabin', 'CabinClass', 'Ticket'], axis=1, inplace=True)
training_data.fillna(training_data.mean(), inplace=True)

#Test Data same operations
test_data['CabinClass'] = test_data['Cabin'].astype(str).str[0]
test_data = pd.concat([test_data, pd.get_dummies(test_data['Sex'], prefix='Sex')], axis =1)
test_data = pd.concat([test_data, pd.get_dummies(test_data['CabinClass'], prefix='Cabin')], axis=1)
test_data.drop(['Cabin_n', 'Sex', 'Name', 'Cabin', 'CabinClass', 'Ticket'], axis=1, inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

X_training = training_data.drop(['PassengerId', 'Survived', 'SibSp', 'Parch', 'Embarked'], axis=1)
y_training = training_data['Survived']
X_training.info()

X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.3, random_state=42)

X_test = test_data.drop(['PassengerId', 'SibSp', 'Parch', 'Embarked'], axis=1)
X_test.info()

#Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_training, y_training)
y_hat = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_hat})
print(output)
output.to_csv('titanic_submission.csv', index=False)
