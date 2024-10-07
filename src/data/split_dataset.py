# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Prompt the user for input file paths
    input_filepath= click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_data = f"{input_filepath}\\raw.csv"
    output_filepath = click.prompt('Enter the file path for the output preprocessed data (e.g., output/preprocessed_data.csv)', type=click.Path())

    # Call the main data processing function with the provided file paths
    process_data(input_filepath_data, output_filepath)

def process_data(input_filepath_data, output_folderpath):
 
    #--Importing dataset
    df = pd.read_csv(input_filepath_data, sep=",")

    # 1. Convertir la colonne en type datetime
    df['date'] = pd.to_datetime(df['date'])

    # 2. Extraire des informations utiles
    df['year'] = df['date'].dt.year        # Année
    df['month'] = df['date'].dt.month      # Mois
    df['day'] = df['date'].dt.day          # Jour
    df['hour'] = df['date'].dt.hour        # Heure
    df['minute'] = df['date'].dt.minute    # Minute



    target = df['silica_concentrate']
    feats = df.drop(columns=['silica_concentrate', 'date'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state = 42)

    # Create folder if necessary 
    if not os.path.exists(output_folderpath):
        os.makedirs(output_folderpath)

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()