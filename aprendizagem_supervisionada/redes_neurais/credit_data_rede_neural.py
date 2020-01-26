from sklearn.neural_network import MLPClassifier
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

clf = MLPClassifier(learning_rate_init = 0.01,
                    max_iter = 1000,
                    tol = 0.000010,
                    verbose = True)

previsores_treinamento, previsores_teste, target_treinamento, target_teste = train_test_split(previsores,
                                                                                              target)

clf.fit(previsores_treinamento, target_treinamento)
result = clf.predict(previsores_teste)

score = accuracy_score(target_teste, result) * 100
matriz = confusion_matrix(target_teste, result)