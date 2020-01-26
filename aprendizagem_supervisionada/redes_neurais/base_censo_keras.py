from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('adult.data')

previsoes = data.iloc[:, 0:14].values
classe = data['income'].values

column = ColumnTransformer([("onehoteencoder", 
                            OneHotEncoder(),
                            [1, 3, 5, 6, 7, 8, 9, 13])],
                           remainder="passthrough")

previsoes = column.fit_transform(previsoes).toarray()

label_encoder = LabelEncoder()
classe = label_encoder.fit_transform(classe)

scale = StandardScaler()
previsoes = scale.fit_transform(previsoes)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsoes, classe, test_size = 0.2)

clf = Sequential()
clf.add(Dense(units = 55, activation = "relu", input_dim = 108))
clf.add(Dense(units = 55, activation = "relu"))
clf.add(Dense(units = 1, activation = "sigmoid"))
clf.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
clf.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)
result = clf.predict(previsores_teste)
result = (result > 0.5)

score = accuracy_score(classe_teste, result) * 100
matriz = confusion_matrix(classe_teste, result)