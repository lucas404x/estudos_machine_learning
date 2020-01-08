from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("plano_saude2.csv")

X = df["idade"].values.reshape(-1, 1)
y = df["custo"].values

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = y.reshape(-1, 1)
y = scaler_y.fit_transform(y)

regre = MLPRegressor(hidden_layer_sizes = (50, 50), max_iter = 1000)

regre.fit(X, y.ravel())
print(regre.score(X, y.ravel()))

predicao = regre.predict(scaler_x.transform([[40]]))
result = scaler_y.inverse_transform((predicao.reshape(-1, 1)))
