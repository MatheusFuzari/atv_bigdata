import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/afw/bigdata/Exercices/aula9/social_network_ads.csv')
df = df.rename(columns={"User ID":"UserId"})
#print(df.head())
#print(df.isnull())
#print(df.isna())
x = df.iloc[:,[2,3]].values #Pega Age e EstimatedSalary
y = df.iloc[:,-1].values #Pega o Purchased
x_treino, x_teste, y_treino, y_teste  = train_test_split(x,y,test_size=0.3,random_state=3)
sc = StandardScaler()
x_treino = sc.fit_transform(x_treino) # normalizar e cirar modelo
x_teste = sc.transform(x_teste) # normalizar apenas
algoritimo = SVC(kernel='linear',random_state=3)
algoritimo.fit(x_treino,y_treino)
y_pred = algoritimo.predict(x_teste)
x_teste_inverse = sc.inverse_transform(x_teste)
#plt.figure(figsize=(10,6))
#plt.scatter(x_teste_inverse[y_pred == 0,0],x_teste_inverse[y_pred == 0,1], c='red', label="Não Compra")
#plt.scatter(x_teste_inverse[y_pred == 1,0],x_teste_inverse[y_pred == 1,1], c='blue', label='Compra')
#plt.title('SVM')
#plt.xlabel("Idade")
#plt.ylabel("Salário Anual")
#plt.legend()
#plt.show()
_x,_y = x_treino, y_treino
x1,x2 = np.meshgrid(np.arange(start = _x[:,0].min()-1,stop= _x[:,0].max()+1, step=0.01),
                    np.arange(start = _x[:,1].min()-1,stop=_x[:,1].max()+1,step=0.01))
plt.contour(x1,x2, algoritimo.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75, cmap = ListedColormap(('red','blue')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(_y)):
    plt.scatter(_x[_y == j,0],_x[_y==j,1],
                c = ListedColormap(('gray','black'))(i),label = j)
plt.show()