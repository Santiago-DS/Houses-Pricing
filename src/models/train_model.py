# -*- coding: utf-8 -*-

###############################
##### IMPORTS
###############################
import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import make_scorer, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

################################
##### CARREGAMENTO DOS DADOS
################################
print('Script de treinamento do modelo!')
print('Carregando os dados...')
df = pd.read_csv(r"../../data/processed/train_casas.csv")



################################
##### DIVISAO DAS BASES
################################
print('Preparando as bases...')
SEED = 42
np.random.seed(SEED)

# dividindo em dados de treino e teste
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    shuffle=True,
    random_state=SEED)
X_train = df_train.drop(columns='SalePrice').astype('float').copy()
y_train = df_train['SalePrice'].astype('float').values
X_test= df_test.drop(columns='SalePrice').astype('float').copy()
y_test= df_test['SalePrice'].astype('float').values

################################
##### PIPELINE DE TREINAMENTO
################################
print('Treinando os modelos...')
scaler = MinMaxScaler()
cv = KFold(n_splits=10, shuffle=True)
#params = {}
models = {
    'DecisionTreeRegressor': DecisionTreeRegressor(), 
    'RandomForestRegressor': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'XGBRegressor': XGBRegressor()}
models_r2score = []
scores = make_scorer(r2_score)

# ! Verifique se a porta 5000 está liberada em seu computador antes de testar
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('houses-pricing-AmericanHouses')

for name, model in zip(list(models.keys()), list(models.values())):
    with mlflow.start_run():
        if model == XGBRegressor():
            mlflow.xgboost.log_model(model, name)
        else:
            mlflow.sklearn.log_model(model, name)

        pipe = Pipeline( [('escalonamento', scaler), ('treinamento', model)] )
        resultados = cross_validate(
            pipe,
            X_train,
            y_train,
            cv = cv,
            return_train_score=True
        )
        model_test = pipe.fit(X_train, y_train)
        predict_test = pipe.predict(X_test)
        r2_score_test_database = r2_score(y_test, predict_test)
        crossval_r2_score_test_medio = resultados['test_score'].mean()
        crossval_r2_score_test_std = resultados['test_score'].std()
        limite_superior = crossval_r2_score_test_medio + 2*crossval_r2_score_test_std
        limite_inferior = crossval_r2_score_test_medio - 2*crossval_r2_score_test_std


        resultados[name] = r2_score_test_database 


        mlflow.log_param("Estimator Name", name)
        mlflow.log_metric('mean_crossval_test_r2_score', crossval_r2_score_test_medio*100)
        mlflow.log_metric('std_database_r2_score_', crossval_r2_score_test_std*100)
        mlflow.log_metric('limite_superior', limite_superior*100)
        mlflow.log_metric('limite_inferior', limite_inferior*100)
        mlflow.log_metric('test_database_r2_score_', r2_score_test_database*100)
        models_r2score.append(r2_score_test_database)
        print(f'Treinamento do modelo {name} concluido!')

models_result = {
    'Modelo': list(models.keys()),
    'r2_score': models_r2score
}

print('\nO melhor modelo foi selecionado com base no r2_score.\nO modelo selecionado foi:')
models_df = pd.DataFrame(models_result)
best_estimator_name = models_df.nlargest(columns='r2_score',n=1).Modelo.values[0]
print(best_estimator_name)
