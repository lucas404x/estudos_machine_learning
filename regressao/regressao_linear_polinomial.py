from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import pandas as pd

data = pd.read_csv("plano_saude2.csv")

X = data["idade"].values
X = X.reshape(-1, 1)
y = data["custo"].values
poly = PolynomialFeatures(degree = 2)

X = poly.fit_transform(X)
regre = LinearRegression()

regre.fit(X, y)

score = regre.score(X, y)