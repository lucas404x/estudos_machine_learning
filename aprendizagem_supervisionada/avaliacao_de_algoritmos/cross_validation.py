from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

import pickle
import pandas as pd
import numpy as np


data = pd.read_csv('credit_data.csv')

media = data.loc[data['age'] > 0, 'age'].mean()
data.loc[data['age'] < 0, 'age'] = media

previsores = data.iloc[:, 1:4].values
classe = data['default'].values

imputer = SimpleImputer()

previsores = imputer.fit_transform(previsores)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
splits = int(input("Splits: "))

clfs = (KNeighborsClassifier(n_neighbors=5), 
        GaussianNB(), 
        DecisionTreeClassifier(criterion = "entropy"), 
        RandomForestClassifier(criterion = "entropy", n_estimators = 100),
        SVC(gamma = "scale"))

matrizes = []
scores = []

for i in range(len(clfs)):
    scores.append([])
    for j in range(30):
        score = 0
        kfold = StratifiedKFold(n_splits = splits, shuffle = True, random_state = j)
        for n_train, n_test in kfold.split(previsores, classe):
            clfs[i].fit(previsores[n_train], classe[n_train])
            result = clfs[i].predict(previsores[n_test])
            matrizes.append(confusion_matrix(classe[n_test], result))
            score += accuracy_score(classe[n_test], result) * 100
        scores[i].append(score/splits)
        
with open("resultados.csv", mode = "w") as file:
    file.write("seed,KNN,Naive_Bayes,Decision_Tree,Random_Forest,SVM\n")
    for seed in range(len(scores[0])):
        file.write(str(seed) + ",")
        for clf in range(len(scores)):
            file.write(str(scores[clf][seed]))
            if clf < len(scores) - 1:
                file.write(",")
        file.write("\n")
    
media_scores = [sum(score)/len(score) for score in scores]
maior_score = max(media_scores)
clf_maior_score = clfs[media_scores.index(maior_score)]
print("Melhor resultado vai para o {}, com um score médio de {}".format(
        clf_maior_score, maior_score))

clf_maior_score.fit(previsores, classe)

pickle.dump(clf_maior_score, open("melhor_classificador.sav", "wb"))

###### Salvar classificador #######

#mean_matriz = np.mean(matrizes, axis = 0)
#score = cross_validate(clf, previsores, target, cv = 10, verbose = 1)

#score = accuracy_score(target_teste, result) * 100
#matriz = confusion_matrix(target_teste, result)