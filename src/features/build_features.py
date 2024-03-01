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


################################
##### CONSTRUCAO DAS FEATURES
################################

# filtrando os atributos numericos por correlacao
# usarei apenas features com correlacao (em valor absoluto) maior que 0.3
corr = casas[num_cols].corr()
num_cols_selected = corr.loc[ abs(corr['SalePrice']) > 0.3,  'SalePrice']
num_cols_selected = num_cols_selected.iloc[ num_cols_selected.index != 'SalePrice' ].index.tolist()

# filtrando os atributos categoricos que tem correlacao com a variavel alvo
# teste ANOVA para identificar as variaveis categoricas que tem relacao com SalePrice
cat_cols2 = cat_cols + ['SalePrice']
cat_cols_selected = []
a = 0.05
for col in cat_cols2:
    categorias = [dados for grupo, dados in casas[cat_cols2].groupby(col)['SalePrice']]
    anova_result = f_oneway(*categorias)

    if (anova_result.pvalue <= a) & (col != 'SalePrice'):
        cat_cols_selected.append(col)

# variaveis selecionadas
variaveis = ['Id'] + cat_cols + num_cols
casas_features = casas[variaveis]

################################
##### ENCODING
################################
