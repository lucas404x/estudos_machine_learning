import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import pandas as pd

data = pd.read_csv('credit_data.csv')

media = data.loc[data['age'] > 0, 'age'].mean()
data.loc[data['age'] < 0, 'age'] = media

previsores = data.iloc[:, 1:4].values
target = data['default'].values

imputer = SimpleImputer()

previsores = imputer.fit_transform(previsores)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

clf = Sequential()
clf.add(Dense(units = 2, activation = "relu", input_dim = 3))
clf.add(Dense(units = 2, activation = "relu"))
clf.add(Dense(units = 1, activation = "sigmoid"))


previsores_treinamento, previsores_teste, target_treinamento, target_teste = train_test_split(previsores,
                                                                                              target)


clf.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
clf.fit(previsores_treinamento, target_treinamento, batch_size = 10, epochs = 100)
result = clf.predict(previsores_teste)
result = (previsores > 0.5)
#score = accuracy_score(target_teste, result) * 100
#matriz = confusion_matrix(target_teste, result)