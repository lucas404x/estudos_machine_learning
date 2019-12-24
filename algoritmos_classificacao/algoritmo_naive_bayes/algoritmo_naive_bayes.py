#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:54:56 2019

@author: lucas404x
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pandas as pd

data = pd.read_csv("risco_credito.csv")

previsores = data.iloc[:, :4].values
classe = data['risco'].values
label_encoder = LabelEncoder()

for i in range(len(previsores[0])):
    previsores[:, i] = label_encoder.fit_transform(previsores[:, i])

#classe = label_encoder.fit_transform(classe)

classificador = GaussianNB()
classificador = classificador.fit(previsores, classe) # cria a tabela de probabilidades
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])
