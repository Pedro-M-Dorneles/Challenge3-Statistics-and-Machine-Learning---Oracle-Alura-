import matplotlib.pyplot as plt
from Tratamento import Y, X_normal


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_normal, Y, test_size= 0.20)


#Balanceamento apenas dos dados de treino com SMOTE
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)


#Utilizando o algoritimo KNN da biblioteca sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
#Procurando o melhor valor para K, e o melhor F1 desse valor
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1')
grid.fit(X_train, Y_train)

print('\nMelhor valor de K:\n', grid.best_params_)
print('\nMelhor Recall score:\n', grid.best_score_)

#Criando o modelo de fato com o melhor K descoberto pelo grid
#Inicializando o algoritimo em uma variavel
knn = grid.best_estimator_
#Ajustando o modelo aos dados
knn.fit(X_train, Y_train)


#Observando as métricas do modelo
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
Y_prev = knn.predict(X_test)
print('\nAcurácia: ', accuracy_score(Y_test, Y_prev))
print('Precisão: ', precision_score(Y_test, Y_prev))
print('Recall: ', recall_score(Y_test, Y_prev))
print('F1 Score: ', f1_score(Y_test, Y_prev))


#Construindo a curva a partir das previsões feitas pelo modelo
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(Y_test, Y_prev, name='KNN')
plt.show()


#Avaliando com a matriz de confusão
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Criando a matriz
matriz_confusao = confusion_matrix(Y_test, Y_prev)
#Visualizando a matriz de confusão
visualizacao = ConfusionMatrixDisplay(matriz_confusao, display_labels=['Deixou o serviço', 'Não deixou o serviço'])
visualizacao.plot()
plt.show()

#Foi considerado que esse modelo é bem ruim, nem vale a pena ajustar hiperparâmetros
