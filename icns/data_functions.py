from os import path
from typing import List

import pandas as pd

from icns.common import Phenotypic, create_phenotype_path, DataScope, \
    create_patient_time_series_path, all_phenotypic_results_path, Institute, Atlas


def read_time_series_means(time_series_path: str) -> pd.Series:
    return read_time_series(time_series_path).mean()


def read_time_series(time_series_path: str) -> pd.DataFrame:
    time_series_data: pd.DataFrame = pd.read_table(time_series_path)
    # Ignore the first two fields: File and	Sub-brick
    return time_series_data.iloc[:, 2:]


def create_training_data(institutes: List[Institute], atlas: Atlas, phenotype_features: List[str], smoothed: bool = False) -> dict:
    adhd_data = dict()

    for institute in institutes:
        institute_data = dict()

        # Read the phenotype file into a data frame
        phenotype_file_path: str = create_phenotype_path(str(institute), DataScope.TRAIN)
        phenotype_df: pd.DataFrame = pd.read_csv(phenotype_file_path)
        phenotype_df[Phenotypic.SCAN_DIR_ID] = phenotype_df[Phenotypic.SCAN_DIR_ID].apply(
            lambda x: f'{x:07d}')
        # Filter the data considering only selected features and target labels
        all_labels: List[str] = [Phenotypic.SCAN_DIR_ID] + phenotype_features + [Phenotypic.DX]
        selected_phenotype_df: pd.DataFrame = phenotype_df[all_labels].set_index(Phenotypic.SCAN_DIR_ID)
        selected_phenotype_df[Phenotypic.GENDER].fillna(method='pad', inplace=True)

        # Process and collect time series files
        for patient_id, phenotypic in selected_phenotype_df.iterrows():
            # Get patient time series
            time_series_path = create_patient_time_series_path(str(institute), patient_id, atlas, DataScope.TRAIN, smoothed)
            if path.exists(time_series_path):
                time_series_data: pd.DataFrame = pd.read_table(time_series_path)
                # Ignore the first two fields: File and	Sub-brick
                time_series_data = time_series_data.iloc[:, 2:]

                # Create patient dictionary
                patient_data = dict()
                patient_data['time_series'] = time_series_data
                patient_data['phenotypic'] = phenotypic

                institute_data[patient_id] = patient_data

        adhd_data[institute] = institute_data

    return adhd_data


def create_adhd_withheld_data(institutes: List[Institute], atlas: Atlas, phenotype_features: List[str],
                              smoothed: bool = False) -> dict:
    adhd_data = dict()

    # There is only one phenotypic file
    phenotype_df: pd.DataFrame = pd.read_csv(all_phenotypic_results_path)
    phenotype_df['ID'] = phenotype_df['ID'].apply(lambda x: f'{x:07d}')

    for institute in institutes:
        institute_data = dict()

        # Select the rows of the institute
        institute_phenotype_df: pd.DataFrame = phenotype_df.loc[
            phenotype_df[Phenotypic.SITE] == institute.get_code()]
        # Filter the data considering only selected features and target labels
        all_labels: List[str] = ['ID'] + phenotype_features + [Phenotypic.DX]
        selected_phenotype_df: pd.DataFrame = institute_phenotype_df[all_labels].set_index('ID')
        selected_phenotype_df[Phenotypic.GENDER].fillna(method='pad', inplace=True)

        # Process and collect time series files
        for patient_id, phenotypic in selected_phenotype_df.iterrows():
            # Get patient time series
            time_series_path = create_patient_time_series_path(str(institute), patient_id, atlas, DataScope.TEST, smoothed)
            if path.exists(time_series_path):
                time_series_data: pd.DataFrame = pd.read_table(time_series_path)
                # Ignore the first two fields: File and	Sub-brick
                time_series_data = time_series_data.iloc[:, 2:]

                # Create patient dictionary
                patient_data = dict()
                patient_data['time_series'] = time_series_data
                patient_data['phenotypic'] = phenotypic

                institute_data[patient_id] = patient_data

        adhd_data[institute] = institute_data

    return adhd_data
