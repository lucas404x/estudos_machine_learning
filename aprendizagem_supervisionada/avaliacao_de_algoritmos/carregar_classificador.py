from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import pickle
import pandas as pd

data = pd.read_csv('credit_data.csv')

media = data.loc[data['age'] > 0, 'age'].mean()
data.loc[data['age'] < 0, 'age'] = media

previsores = data.iloc[:, 1:4].values
classe = data['default'].values

imputer = SimpleImputer()

previsores = imputer.fit_transform(previsores)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

clf = pickle.load(open("melhor_classificador.sav", "rb"))
print(clf.score(previsores, classe))