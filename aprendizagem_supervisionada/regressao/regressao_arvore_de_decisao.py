from sklearn.tree import DecisionTreeRegressor
import pandas as pd

data = pd.read_csv("plano_saude2.csv")

X = data["idade"].values
X = X.reshape(-1, 1)
y = data["custo"].values
regre = DecisionTreeRegressor()
regre.fit(X, y)
score = regre.score(X, y)