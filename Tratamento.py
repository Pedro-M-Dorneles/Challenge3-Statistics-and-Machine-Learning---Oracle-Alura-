import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#Lendo os dados
dados = pd.read_csv('G:/Meu Drive/Oracle_alura/Estatistica_ML/Challenge_3/dados/dados_tratados.csv')
print(dados.info())

#Removendo valores nulos
dados.fillna(dados.mean(numeric_only=True), inplace=True)

#Removendo colunas inuteis
dados.drop(columns='customerID', inplace=True)

#Realizando o encoding das variaveis
print(dados.nunique())

#Convertendo as variaveis explicativas binarias
dados['Churn'].replace({'No': 0, 'Yes': 1},inplace=True)
dados['customer.gender'].replace({'Female': 0, 'Male': 1}, inplace=True)
dados['customer.Partner'].replace({'No': 0, 'Yes': 1}, inplace=True)
dados['customer.Dependents'].replace({'No': 0, 'Yes': 1}, inplace=True)
dados['phone.PhoneService'].replace({'No': 0, 'Yes': 1}, inplace=True)
dados['account.PaperlessBilling'].replace({'No': 0, 'Yes': 1}, inplace=True)

#Criando uma coluna pra quem tem ou não serviço se phone ou internete, transformando as originais em binarias
dados['has.PhoneService'] = dados['phone.MultipleLines'].map(lambda x: 0 if x=='No phone service' else 1)
dados['phone.MultipleLines'].replace({'No': 0, 'Yes': 1, 'No phone service': 0},inplace=True)

colunas = ['internet.OnlineSecurity','internet.OnlineBackup','internet.DeviceProtection','internet.TechSupport','internet.StreamingTV','internet.StreamingMovies']
dados[colunas]=dados[colunas].replace({'No': 0, 'Yes': 1, 'No internet service': 0})
dados['has.InternetService'] = dados[colunas].apply(lambda linha: 0 if all(valor == 0 for valor in linha) else 1, axis=1)
print(dados.info())


#Convertendo as explicativas nao binarias utilizando o One hot encoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

#Separando variaveis explicativas da independente
X  = dados.drop(columns='Churn')
Y  = dados['Churn']
#Salvando as colunas originais
colunas = X.columns
one_hot = make_column_transformer((OneHotEncoder(drop='if_binary'),['account.Contract','account.PaymentMethod','internet.InternetService']),remainder='passthrough',sparse_threshold=0)

#Aplicando a transformação nas variaveis explicativas
X = one_hot.fit_transform(X)
X = pd.DataFrame(X, columns=one_hot.get_feature_names_out(colunas))
print(Y)
print(X.info())

#Observando a proporção de evasão da variavel dependente
print(Y.value_counts(normalize=True))
#73.5% não saiu, 26.5% saiu


#Realizando a análise da correlação dos parâmetros
correlacoes = {}
for coluna in X.columns:
    correlacoes[coluna] = pd.Series(X[coluna]).corr(Y)

# Convertendo para DataFrame para facilitar a visualização
correlacoes_df = pd.DataFrame.from_dict(correlacoes, orient='index', columns=['Correlação com Churn'])
correlacoes_df = correlacoes_df.sort_values(by='Correlação com Churn', ascending=False)

# Plotando as correlações
plt.figure(figsize=(10, 8))
sns.barplot(x=correlacoes_df['Correlação com Churn'], y=correlacoes_df.index, palette='coolwarm')
plt.title('Correlação das variáveis com Churn')
plt.xlabel('Correlação')
plt.ylabel('Variável')
plt.tight_layout()
#plt.show()


#Fazendo dados normalizados
from sklearn.preprocessing import MinMaxScaler
#Inicializando o objeto da normalização em uma variável
normalizacao = MinMaxScaler()
colunas = X.columns

#Aplicando a transformação nos dados de treinamento com a função .fit_transform()
X_normal = pd.DataFrame(normalizacao.fit_transform(X), columns=colunas)

#Visualizando as informações normalizadas
fig = px.histogram(X_normal, x='onehotencoder__account.Contract_Month-to-month', text_auto=True, color=Y, barmode='group')
#fig.show()