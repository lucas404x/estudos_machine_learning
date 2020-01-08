from apyori import apriori

import numpy as np
import pandas as pd


dados = pd.read_csv("mercado.csv", header = None)
transacoes = []

for i in dados.values:
    i = list(i)
    for j in range(i.count(np.nan)):
        i.remove(np.nan)
    transacoes.append(i)

resultados = list(apriori(transacoes)) # lista com as regras