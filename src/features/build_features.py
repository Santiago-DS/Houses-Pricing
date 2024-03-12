###############################
##### IMPORTS
###############################
import pandas as pd
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder


################################
##### CARREGAMENTO DOS DADOS
################################

# carregamento da base
casas = pd.read_csv(r"../../data/processed/processed_casas.csv")

# separacao das variaveis categoricas e numericas
num_cols = list(casas.drop(columns='Id').select_dtypes(['int', 'float']).columns)
cat_cols = list(casas.select_dtypes('object').columns)


###############################
##### GERANDO DUMMIES
###############################

# gerando contagem das variaveis pela quantidade de categorias unicas
count_cat = casas[cat_cols].describe().T.sort_values('unique', ascending=False)['unique'].reset_index()
two_cats = count_cat.loc[count_cat['unique'] < 3, 'index'].values # quero apenas as que tenham menos 3 categorias unicas

# gerando as dummies para as variaveis com mais de 3 categorias
cat_dummies = pd.get_dummies(casas[cat_cols].drop(columns=two_cats))
two_cat_dummies = pd.get_dummies(casas[two_cats], drop_first=True)

# juntando as duas
casas_dummies = pd.concat([casas[num_cols], cat_dummies, two_cat_dummies])