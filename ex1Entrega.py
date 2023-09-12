import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn import tree
import matplotlib.pylab as plt
import seaborn as sns

vel = np.array([10,5,8,6,12,4]),
complex=np.array(['baixa','alta','media','alta','baixa','alta']),
manutencao=np.array(['baixa','alta','media','alta','baixa','media']),
classif=np.array(['montagem','teste','montagem','teste','monstagem','teste'])    

x_treino, x_teste, y_treino, y_teste = train_test_split(vel.reshape(-1,1),complex.reshape(-1,1),manutencao.reshape(-1,1),classif.reshape(-1,1),test_size=0.2,random_state=42)
model = DecisionTreeClassifier()
model.fit(x_treino,y_treino)
prev = model.predict(x_teste)
acurracy = accuracy_score(y_teste,prev)
fig = plt.figure(figsize=(10,8))
tree.plot_tree(model,feature_names=vel.tolist(),class_names=classif.tolist(),filled=True)
plt.show()

sns.boxplot(x=classif,y=vel)
sns.set(font_scale=1)
plt.grid(True)
plt.xlabel('Classes')
plt.ylabel('Notas')
plt.show()


sns.barplot(x=classif,y=vel)
sns.set(font_scale=1)
plt.grid(True)
plt.xlabel('Classes')
plt.ylabel('Notas')
plt.show()

