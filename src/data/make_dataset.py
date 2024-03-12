# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
#from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    # esse script deve ser executado por linha de comando da seguinte forma: python seu_script.py caminho_para_o_arquivo_de_entrada.csv caminho_para_o_arquivo_de_saida.csv

    # leitura da base
    df_casas_raw = pd.read_csv(r'{}'.format(input_filepath))

    ######################## TRATAMENTO DOS NULOS   
    # preenchimento dos nulos
    num_cols = df_casas_raw.select_dtypes(['float', 'int']).columns
    obj_cols = df_casas_raw.drop(columns='Electrical').select_dtypes('object').columns
    df_casas_raw[num_cols] = df_casas_raw[num_cols].fillna(0)
    df_casas_raw[obj_cols] = df_casas_raw[obj_cols].fillna('NotAvailable')

    # excluindo linhas nulas
    df_casas_raw = df_casas_raw.loc[ ~(df_casas_raw['Electrical'].isnull() )]
    ########################
    
    # salvando os resultados
    df_casas_raw.to_csv(r'{}'.format(output_filepath), index=False)
    

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()
