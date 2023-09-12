import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn import tree
import matplotlib.pylab as plt
import seaborn as sns

nota = np.array([8,6,5,9,4,3,7,2,1,5,6,9,8])
classes = np.array(['bom','ruim','ruim','bom','pessimo','pessimo','bom','pessimo','pessimo','ruim','ruim','bom','bom'])

nota_treino, nota_teste, classes_treino,classes_teste = train_test_split(nota.reshape(-1,1),classes.reshape(-1,1),test_size=0.2,random_state=42)
modelo = DecisionTreeClassifier()
modelo.fit(nota_treino,classes_treino)
previsao = modelo.predict(nota_teste)
acurracy = accuracy_score(classes_teste,previsao)
fig = plt.figure(figsize=(10,8))
tree.plot_tree(modelo,feature_names=nota.tolist(),class_names=classes.tolist(),filled=True)
plt.show()

sns.boxplot(x=classes,y=nota)
sns.set(font_scale=1)
plt.grid(True)
plt.xlabel('Classes')
plt.ylabel('Notas')
plt.show()


sns.barplot(x=classes,y=nota)
sns.set(font_scale=1)
plt.grid(True)
plt.xlabel('Classes')
plt.ylabel('Notas')
plt.show()

