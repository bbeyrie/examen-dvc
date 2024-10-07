# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.preprocessing import StandardScaler
import os

@click.command()
@click.argument('input_folder_path', type=click.Path(exists=False), required=0)

def main(input_folder_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Prompt the user for input file paths
    input_folder_path = click.prompt('Enter the folder path for the input data', type=click.Path(exists=True))

    # Call the main data processing function with the provided file paths
    process_data(input_folder_path)

def process_data(input_folder_path):

    X_train_path = f"{input_folder_path}\\X_train.csv"
    X_test_path = f"{input_folder_path}\\X_test.csv"
 
    #--Importing dataset
    X_train = pd.read_csv(X_train_path, sep=",")
    X_test = pd.read_csv(X_test_path, sep=",")

    # Scaler
    scaler = StandardScaler()
    features = X_train.columns
    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features] = scaler.transform(X_test[features])

    # Create folder if necessary 
    if not os.path.exists(input_folder_path):
        os.makedirs(input_folder_path)

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(input_folder_path, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()