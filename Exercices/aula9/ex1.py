import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

data = {"Potência(kW)":np.array([120,80,100,110,90,130,70,105,115,85]),
        "Eficiência(%)":np.array([92,65,75,85,68,95,60,80,88,70]),
        "Idade(anos)":np.array([3,8,5,4,7,2,10,6,3,9]),
        "Tamanho(m²)":np.array([50,45,55,60,48,62,40,58,56,47]),
        "Consumo De Energia":np.array([1,0,0,1,0,1,0,1,1,0])}
df = pd.DataFrame(data=data)
x = df.iloc[:,[0,1]].values
y = df.iloc[:,-1].values
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,random_state=3,test_size=0.3)
sc = StandardScaler()
x_treino = sc.fit_transform(x_treino)
x_teste = sc.transform(x_teste)
algoritimo = SVC(kernel='linear',random_state=3)
algoritimo.fit(x_treino,y_treino)
y_pred = algoritimo.predict(x_teste)
x_teste_inverse = sc.inverse_transform(x_teste)
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