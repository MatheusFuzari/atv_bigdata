import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree

iris = datasets.load_iris()
data = iris.data
target = iris.target

x,y = data, target
classificador = tree.DecisionTreeClassifier()
classificador.fit(x,y)
fig = plt.figure(figsize=(30,25))
tree.plot_tree(classificador, feature_names=iris.feature_names, class_names=iris.target_names.tolist(), filled=True)
plt.savefig('caralho')
a=plt.figure(figsize=(15,10))
plt.scatter(iris.data[:,0], iris.data[:,2],c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[2])
plt.tight_layout()
b = plt.figure(figsize=(15,10))
plt.scatter(iris.data[:,2], iris.data[:,3],c=iris.target)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()
