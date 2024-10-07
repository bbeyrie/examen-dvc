# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import click
import logging
from sklearn.ensemble import RandomForestRegressor
import os

@click.command()
@click.argument('input_data_path', type=click.Path(exists=False), required=0)
@click.argument('input_params_path', type=click.Path(exists=False), required=0)

def main(input_data_path, input_params_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Prompt the user for input file paths
    input_data_path = click.prompt('Enter the folder path for the input data', type=click.Path(exists=True))
    input_params_path = click.prompt('Enter the folder path for the best model params', type=click.Path(exists=True))

    # Call the main data processing function with the provided file paths
    process_data(input_data_path, input_params_path)

def process_data(input_data_path, input_params_path):

    X_train_path = f"{input_data_path}\\X_train_scaled.csv"
    y_train_path = f"{input_data_path}\\y_train.csv"
 
    #--Importing dataset
    X_train = pd.read_csv(X_train_path, sep=",")
    y_train = pd.read_csv(y_train_path, sep=",")

    best_params = joblib.load(os.path.join(input_params_path, f'best_rf_params.pkl'))

    # 1. Définir le modèle de base
    rf = RandomForestRegressor(**best_params)

    #--Train the model
    rf.fit(X_train, y_train.values.ravel())

    output_filepath = os.path.join(input_params_path, f'model_rf.pkl')
    joblib.dump(rf, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()