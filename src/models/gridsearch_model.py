# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import click
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import os

@click.command()
@click.argument('input_folder_path', type=click.Path(exists=False), required=0)
@click.argument('output_folder_path', type=click.Path(exists=False), required=0)

def main(input_folder_path, output_folder_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Prompt the user for input file paths
    input_folder_path = click.prompt('Enter the folder path for the input data', type=click.Path(exists=True))
    output_folder_path = click.prompt('Enter the folder path for the best model params', type=click.Path(exists=True))

    # Call the main data processing function with the provided file paths
    process_data(input_folder_path, output_folder_path)

def process_data(input_folder_path, output_folder_path):

    X_train_path = f"{input_folder_path}\\X_train_scaled.csv"
    X_test_path = f"{input_folder_path}\\X_test_scaled.csv"
    y_train_path = f"{input_folder_path}\\y_train.csv"
    y_test_path = f"{input_folder_path}\\y_test.csv"
 
    #--Importing dataset
    X_train = pd.read_csv(X_train_path, sep=",")
    X_test = pd.read_csv(X_test_path, sep=",")
    y_train = pd.read_csv(y_train_path, sep=",")
    y_test = pd.read_csv(y_test_path, sep=",")

    # 1. Définir le modèle de base
    rf = RandomForestRegressor(random_state=42)

    # 2. Définir la grille d'hyperparamètres à tester
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
        'bootstrap': [True, False]
    }

    # 3. Configurer le GridSearch avec validation croisée (cv=5)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # 4. Entraîner le modèle avec GridSearch sur l'ensemble d'entraînement
    grid_search.fit(X_train, y_train.values.ravel())

    best_params = grid_search.best_params_

    # Create folder if necessary 
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_filepath = os.path.join(output_folder_path, f'best_rf_params.pkl')
    joblib.dump(best_params, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]


    main()