from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import pandas as pd

base = pd.read_csv("credit_card_clients.csv", header = 1)

base["BILL_TOTAL"] = base["BILL_AMT1"] + base["BILL_AMT2"] + base["BILL_AMT3"] + \
                     base["BILL_AMT5"] + base["BILL_AMT5"] + base["BILL_AMT6"]

scaler = StandardScaler()
X = scaler.fit_transform(base.iloc[:, [1, 25]].values)

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
previsoes = kmeans.predict(X)
