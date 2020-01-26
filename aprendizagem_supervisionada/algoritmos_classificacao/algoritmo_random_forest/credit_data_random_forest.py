#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:19:31 2019

@author: lucas404x


Usar a padronização não alterouo resultado de forma
significativa, isto porque, o algoritmo arvore de decisão não precisa usar 
métodos de padronização e/ou normalização.
"""

from sklearn.metrics import accuracy_score, confusion_matrix # mede o quanto ele acertou
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier, export # classificador arvore de decisao
# from sklearn.naive_bayes import GaussianNB # classificador naive bayes
from sklearn.model_selection import train_test_split # gera dados de teste e de treino
from sklearn.impute import SimpleImputer # troca os valores nulos a partir de uma estrategia
from sklearn.preprocessing import StandardScaler # escalona os valores
import pandas as pd
import numpy as np

data = pd.read_csv('credit_data.csv')

# trocando os valores negativos pela média

media = data.loc[data['age'] > 0, 'age'].mean()
data.loc[data['age'] < 0, 'age'] = media

for column in data.columns:
    print("Column: " + column)
    print(True in pd.isnull(data[column]))
    print("**************")

previsores = data.iloc[:, 1:4].values
attr_classe = data['default'].values

# trocando os valores nulos pela média

impute = SimpleImputer(missing_values=np.nan, strategy="mean")
impute = impute.fit(previsores)
previsores = impute.transform(previsores)

"""
devemos mudar a escala dos valores,
pois a diferença de um número para o outro é muito alta!
"""

# ele usa a padronização para ajustar a escala dos valores
scaler = StandardScaler()

# ele ajusta os valores e os aplica
previsores = scaler.fit_transform(previsores)

# dividindo a base dados em testes e treinamento
previsores_treinamento, previsores_teste, attr_classe_treinamento, attr_classe_teste = train_test_split(previsores, 
                                                                                                        attr_classe, 
                                                                                     test_size = 0.3,
                                                                                                        random_state = 0)
clf = RandomForestClassifier(n_estimators = 20, criterion = "entropy", random_state = 0)
clf.fit(previsores_treinamento, attr_classe_treinamento)
resultado = clf.predict(previsores_teste)
acuracia = accuracy_score(attr_classe_teste, resultado) * 100
matriz = confusion_matrix(attr_classe_teste, resultado)
