from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

data = pd.read_csv('credit_data.csv')

media = data.loc[data['age'] > 0, 'age'].mean()
data.loc[data['age'] < 0, 'age'] = media

previsores = data.iloc[:, 1:4].values
classe = data['default'].values

imputer = SimpleImputer()

previsores = imputer.fit_transform(previsores)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

score = 0
splits = int(input("Splits: "))
matrizes = []

kfold = StratifiedKFold(n_splits = splits, shuffle = True, random_state = 42)

for n_train, n_test in kfold.split(previsores, classe):
    clf = KNeighborsClassifier()
    clf.fit(previsores[n_train], classe[n_train])
    result = clf.predict(previsores[n_test])
    matrizes.append(confusion_matrix(classe[n_test], result))
    score += accuracy_score(classe[n_test], result) * 100

mean_test = score/splits 
mean_matriz = np.mean(matrizes, axis = 0)
#score = cross_validate(clf, previsores, target, cv = 10, verbose = 1)

#score = accuracy_score(target_teste, result) * 100
#matriz = confusion_matrix(target_teste, result)