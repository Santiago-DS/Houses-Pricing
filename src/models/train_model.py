# -*- coding: utf-8 -*-

###############################
##### IMPORTS
###############################
import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold

from sklearn.metrics import make_scorer, r2_score
from sklearn.tree import DecisionTreeRegressor

################################
##### CARREGAMENTO DOS DADOS
################################
df = pd.read_csv(r"../../data/processed/train_casas.csv")

# divisao em X e y
X = df.drop(columns='SalePrice').astype('float').copy()
y = df['SalePrice'].astype('float').values


# criando a pipeline
SEED = 42
np.random.seed(SEED)
scaler = MinMaxScaler()
cv = KFold(n_splits=10, shuffle=True)
params = {}
model = DecisionTreeRegressor()
scores = make_scorer(r2_score)

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('houses-pricing-AmericanHouses')

with mlflow.start_run():
    mlflow.sklearn.autolog()
    pipe = Pipeline( [('escalonamento', scaler), ('treinamento', model)] )
    resultados = cross_validate(
        pipe,
        X,
        y,
        cv = cv,
        return_train_score=True
    )

    mlflow.log_metric('r2_score', resultados['test_score'].mean()*100)
