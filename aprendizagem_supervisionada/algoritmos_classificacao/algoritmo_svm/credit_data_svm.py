from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import pandas as pd

data = pd.read_csv('credit_data.csv')

media = data.loc[data['age'] > 0, 'age'].mean()
data.loc[data['age'] < 0, 'age'] = media

previsores = data.iloc[:, 1:4].values
target = data['default'].values

imputer = SimpleImputer()

previsores = imputer.fit_transform(previsores)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

previsores_treinamento, previsores_teste, target_treinamento, target_teste = train_test_split(previsores,
                                                                                              target,
                                                                                              random_state = 0)

clf = SVC(C = 2.0, gamma = "scale")
clf.fit(previsores_treinamento, target_treinamento)
result = clf.predict(previsores_teste)

score = accuracy_score(target_teste, result) * 100
matriz = confusion_matrix(target_teste, result)