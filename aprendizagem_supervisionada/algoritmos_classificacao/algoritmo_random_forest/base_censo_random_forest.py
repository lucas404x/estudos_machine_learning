# -*- coding: utf-8 -*-
"""
Lições extras:
    
Nem sempre é bom aplicar tudo, deve ser sempre testado para ver os resultados.
Aplicar o escalonamento nas variaveis Dummy piora os resultados e, portanto,
o recomendando é usar ou o escolonamento ou converter as variaveis em Dummy.
"""

# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('adult.data')

previsoes = data.iloc[:, 0:14].values
classe = data['income'].values

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

for i in range(len(previsoes[0])):
    previsoes[:, i] = label_encoder.fit_transform(previsoes[:, i])

scale = StandardScaler()
previsoes = scale.fit_transform(previsoes)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsoes,
                                                                                              classe,
                                                                                              random_state = 0,
                                                                                              test_size = 0.2)
clf = RandomForestClassifier(n_estimators = 30, criterion = "entropy", random_state = 0)

clf.fit(previsores_treinamento, classe_treinamento)
result = clf.predict(previsores_teste)

score = accuracy_score(classe_teste, result) * 100
matriz = confusion_matrix(classe_teste, result)
