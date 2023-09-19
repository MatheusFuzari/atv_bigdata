import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit # Divide de forma simetria o df????
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
def validador(x,y):
    validador = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
    for treino_id, teste_id in validador.split(x,y):
        x_train, x_test = x[treino_id], x[teste_id]
        y_train, y_teste = y[treino_id], y[teste_id]
    return x_train,x_test,y_train,y_teste
def executar_class(classificador,x_test,x_train,y_train):
    arvore = classificador.fit(x_train,y_train)
    y_pred = arvore.predict(x_test)
    return y_pred
def salvar_arvore(classificador, nome):
    plt.figure(figsize=(300,100))
    tree.plot_tree(classificador,filled=True,fontsize=14)
    plt.savefig(nome)
    plt.close()
def validar_arvore(y_test,y_pred):
    print(confusion_matrix(y_pred,y_test))
    accuracy_score(y_test, y_pred.round(),normalize=False)
df = pd.read_csv('./creditcard.csv',sep=',')
#print(df.head(10)) # retorna as 10 primeiras linhas
#print(df.info()) # retorna o tipo de dado das colunas
#print(df.describe())

numTransacoes = df['Class'].count() # retorna a quantidade de linhas
numFraude = df['Class'].sum() # retorna onde a class é != de 0
numNormais = numTransacoes-numFraude # realiza o calculo total menos a soma dos != 0
print(numTransacoes) # total
print(numFraude) # fraude
print(numNormais) # normal
fraudePerc = (numFraude / numTransacoes)*100
print(fraudePerc)
normalPerc = (numNormais/numTransacoes)*100
print(normalPerc)

x = df.drop('Class', axis=1).values # axis = 1 (coluna) || axis = 0 (linha)
y = df['Class'].values
x_train,x_test,y_train,y_test = validador(x,y)
classificador_arvore_decisao = tree.DecisionTreeClassifier()
arvore = classificador_arvore_decisao.fit(x_train,y_train)
y_pred = executar_class(classificador_arvore_decisao,x_test,x_train,y_train)
salvar_arvore(classificador_arvore_decisao,'xereca')
validar_arvore(y_test,y_pred)
print(precision_score(y_pred,y_test))
print(recall_score(y_pred,y_test))
print(f1_score(y_pred,y_test))