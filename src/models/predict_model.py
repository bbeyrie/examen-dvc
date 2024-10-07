# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import click
import logging
import os
from sklearn.metrics import mean_squared_error, r2_score


@click.command()
@click.argument('input_data_path', type=click.Path(exists=False), required=0)
@click.argument('input_models_path', type=click.Path(exists=False), required=0)

def main(input_data_path, input_models_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Prompt the user for input file paths
    input_data_path = click.prompt('Enter the folder path for the input data', type=click.Path(exists=True))
    input_models_path = click.prompt('Enter the folder path for the model', type=click.Path(exists=True))

    # Call the main data processing function with the provided file paths
    process_data(input_data_path, input_models_path)

def process_data(input_data_path, input_models_path):

    X_test_path = f"{input_data_path}\\X_test_scaled.csv"
    y_test_path = f"{input_data_path}\\y_test.csv"
 
    #--Importing dataset
    X_test = pd.read_csv(X_test_path, sep=",")
    y_test = pd.read_csv(y_test_path, sep=",")

    output_filepath = os.path.join(input_models_path, f'model_rf.pkl')

    # Définir le modèle de base
    rf = joblib.load(output_filepath)

    y_pred = pd.DataFrame(data=rf.predict(X_test), columns=['silica_concentrate'])

    # Évaluer le modèle avec l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Mean Squared Error': mse,
        'R2 Score': r2
    }

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([y_pred], ['y_pred']):
        output_filepath = os.path.join(input_data_path, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

    # Create folder if necessary 
    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    #--Saving the metrics
    with open("metrics/scores.json", 'w') as json_file:
        json.dump(metrics, json_file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()