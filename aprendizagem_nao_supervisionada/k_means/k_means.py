from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import random
import numpy as np
import matplotlib.pyplot as plt


x = [random.randint(18, 70) for x in range(15)]
y = [int(random.random() * 10000) for x in range(15)]

scaler = StandardScaler()
base = np.array([[a, b] for a, b in zip(x, y)])
base = scaler.fit_transform(base)

kmeans = KMeans(n_clusters = 3, verbose = 1)
kmeans.fit(base)
centroides = kmeans.cluster_centers_
rotulos = kmeans.labels_

cores = ("g.", "r.", "b.")

for i in range(len(x)):
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]])
