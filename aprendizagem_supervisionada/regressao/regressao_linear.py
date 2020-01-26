from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_csv("house_prices.csv")

X = df.iloc[:, 2:19].values
y = df["price"].values


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 0)

regre = LinearRegression()

regre.fit(X_train, y_train)

score = regre.score(X_train, y_train)