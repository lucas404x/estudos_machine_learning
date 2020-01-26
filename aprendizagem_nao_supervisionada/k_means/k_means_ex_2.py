from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

x, y = make_blobs(n_samples = 400, centers = 3)

plt.scatter(x[:, 0], x[:, 1])

kmeans = KMeans(n_clusters = 3)
kmeans.fit(x)
previsoes = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c = previsoes)
