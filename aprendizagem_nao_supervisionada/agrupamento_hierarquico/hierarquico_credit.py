from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

import pandas as pd

base = pd.read_csv("credit_card_clientes.csv", header = 1)

base["BILL_TOTAL"] = base["BILL_AMT1"] + base["BILL_AMT2"] + base["BILL_AMT3"] + \
                     base["BILL_AMT5"] + base["BILL_AMT5"] + base["BILL_AMT6"]

scaler = StandardScaler()
X = scaler.fit_transform(base.iloc[:, [1, 25]].values)

dendograma = dendrogram(linkage(X, method = "ward"))
