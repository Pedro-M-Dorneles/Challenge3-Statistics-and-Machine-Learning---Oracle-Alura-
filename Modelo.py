import pandas as pd
import matplotlib.pyplot as plt
from Tratamento import X, Y


#Realizando a divisão dos dados com 80% usado para treino
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.20)


#Balanceamento apenas dos dados de treino com SMOTE
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, Y_train)


#Criando o modelo de arvore e utilizando a validação cruzada
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

#Criando um modelo
floresta = RandomForestClassifier(max_depth = 10, random_state=5)
#Aplicando o floresta
floresta.fit(X_train, Y_train)
print('\nDesempenho do modelo' ,floresta.score(X_test, Y_test))


#Validação cruzada com pipeline
from imblearn.pipeline import Pipeline as imbpipeline

pipeline = imbpipeline([
    ('smote', SMOTE(random_state=5)),
    ('modelo', RandomForestClassifier(max_depth=10, random_state=5))
])
skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=5)
cv_resultados = cross_validate(pipeline, X, Y, cv=skf, scoring='recall')
print('\nResultados da validação cruzada\n', cv_resultados)

#Criando uma função para obter o intervalo de confiança
def intervalo_conf(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'\nIntervalo de confiança: [{media - 2*desvio_padrao}, {min(media + 2*desvio_padrao, 1)}]')
intervalo_conf(cv_resultados)


#Observando as métricas do modelo
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
Y_prev = floresta.predict(X_test)
print('Acurácia: ', accuracy_score(Y_test, Y_prev))
print('Precisão: ', precision_score(Y_test, Y_prev))
print('Recall: ', recall_score(Y_test, Y_prev))
print('F1 Score: ', f1_score(Y_test, Y_prev))

#Construindo a curva a partir das previsões feitas pelo modelo
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(Y_test, Y_prev, name='Arvore de decisão')
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


#Entendendo agora as features mais importantes do modelo de floresta, as 10 primeiras
from yellowbrick.model_selection import FeatureImportances
viz =  FeatureImportances(floresta, relative=False)#O parâmetro relative=False define a importancia absoluta
viz.fit(X_train, Y_train)
viz.show()

importances = floresta.feature_importances_
#Transformando em um data frame e ordenando os valores para melhorar a visualização
feature_importances = pd.DataFrame({'Features': X.columns, 'Importances': importances})
feature_importances.sort_values('Importances', ascending=False, inplace=True)
print('Importancia de cada feature',feature_importances)


# Função para calcular as métricas de classificação
def calcular_metricas_classificacao(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metricas = {
        'Acurácia': round(acc, 4),
        'Precisão': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4)
    }
    return metricas

# Redeterminando o modelo com base nas features mais importantes
# Criando um data frame onde os índices são as métricas
results_df = pd.DataFrame(index=['Acurácia', 'Precisão', 'Recall', 'F1 Score'])

# Recriando o modelo da floresta
model_selected_features = RandomForestClassifier(random_state=5, max_depth=5)

# Criando um loop, para pegar as 1 a 19 features mais importantes
for count in range(1, 20):
    # Selecionando os nomes das features com base no data frame das features mais importantes
    selected_features = feature_importances['Features'].values[:count]
    # Pegando os dados de treino e teste de acordo com as features selecionadas
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    # Ajustando o modelo com esses novos dados de treino
    model_selected_features.fit(X_train_selected, Y_train)
    # Fazendo a previsão do modelo com base nos novos dados de teste
    Y_pred = model_selected_features.predict(X_test_selected)
    # Calculando as novas métricas
    metricas = calcular_metricas_classificacao(Y_test, Y_pred)
    # Colocando o resultado das métricas em um data frame
    results_df[count] = list(metricas.values())
print('\nResultados:\n', results_df)


#Mantendo apenas as 17 primeiras features
selected_features = feature_importances['Features'].values[:17]
X_select_features = X[selected_features]


#Separando os dados de treino e teste apenas com essas features
X_train, X_test, Y_train, Y_test = train_test_split(X_select_features, Y, random_state=5, test_size=0.20)

#Recriando o pipeline
pipeline = imbpipeline([
    ('smote', SMOTE(random_state=5)),
    ('modelo', RandomForestClassifier(random_state=5, class_weight={0: 2, 1: 1}))
])

#Fazendo o modelo só que com os hiperparâmetros
#Grid dos parâmetros do RandomForestClassifier
param_grid = {
    'modelo__n_estimators': [100, 150, 200],
    'modelo__max_depth': [5, 10, 15],
    'modelo__min_samples_split': [2, 4, 6],
    'modelo__min_samples_leaf': [1, 2, 3]
}

#Importando o GridSearchCV e criando o modelo
from sklearn.model_selection import GridSearchCV
floresta_grid = GridSearchCV(pipeline, param_grid=param_grid, scoring='recall', cv=skf)

#Treinando o modelo
floresta_grid.fit(X_train, Y_train)
#Visualizando os melhores parâmetros com o metodo .best_params_
print(floresta_grid.best_params_)


#Observando novamente as métricas
Y_prev = floresta_grid.predict(X_test)
print('Acurácia: ', accuracy_score(Y_test, Y_prev))
print('Precisão: ', precision_score(Y_test, Y_prev))
print('Recall: ', recall_score(Y_test, Y_prev))
print('F1 Score: ', f1_score(Y_test, Y_prev))

#visualizando a matriz de confusão de novo
matriz_confusao = confusion_matrix(Y_test, Y_prev)
visualizacao = ConfusionMatrixDisplay(matriz_confusao, display_labels=['Deixou o serviço', 'Não deixou o serviço'])
visualizacao.plot()
plt.show()