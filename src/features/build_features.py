###############################
##### IMPORTS
###############################
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore")

################################
##### CARREGAMENTO DOS DADOS
################################

# carregamento da base
casas = pd.read_csv(r"../../data/processed/processed_casas.csv")
casas = casas.drop(columns=['MoSold', 'YrSold']) 
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
casas_dummies = pd.concat([casas[num_cols], cat_dummies, two_cat_dummies], axis=1)


##############################################################
##### TRATANDO COLINEARIDADE DAS VARIAVEIS INDEPENDENTES #####
##############################################################

# convertendo booleano para inteiro
X = casas_dummies.drop(columns='SalePrice')
booleans_cols = X.select_dtypes('bool').columns
X[booleans_cols] = X[booleans_cols].astype('int')

# Usando o metodo VIR
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]

# variaveis sem colinearidade
features_no_corr = vif_data.loc[vif_data['VIF']<=5].feature.values
X = X[features_no_corr]