from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

import numpy as np
import random

x = [int(random.random() * 10000) for x in range(15)]
y = [random.randint(18, 80) for x in range(15)]

scaler = StandardScaler()
base = scaler.fit_transform(np.array([[a, b] for a, b in zip(x, y)]))

                            
dendograma = dendrogram(linkage(base, method = "ward"))

hc = AgglomerativeClustering(n_clusters = 3)
result = hc.fit_predict(base) 