from os import path
from typing import List

import pandas as pd

from icns.common import PhenotypeLabels, create_phenotype_path, DataScope, \
    create_patient_time_series_path


def time_series_mean(time_series_path: str) -> pd.Series:
    time_series_data: pd.DataFrame = pd.read_table(time_series_path)
    # Ignore the first two fields: File and	Sub-brick
    time_series_filtered: pd.DataFrame = time_series_data.iloc[:, 2:]
    # Apply mean
    return time_series_filtered.mean()


def create_training_data(institutes: List[str], atlas: str, phenotype_features: List[str]) -> pd.DataFrame:
    for institute in institutes:
        # Read the phenotype file into a data frame
        phenotype_file_path: str = create_phenotype_path(institute, DataScope.TRAIN)
        phenotype_df: pd.DataFrame = pd.read_csv(phenotype_file_path)
        phenotype_df[PhenotypeLabels.SCAN_DIR_ID] = phenotype_df[PhenotypeLabels.SCAN_DIR_ID].apply(
            lambda x: f'{x:07d}')
        # Filter the data considering only selected features and target labels
        all_labels: List[str] = [PhenotypeLabels.SCAN_DIR_ID] + phenotype_features + [PhenotypeLabels.DX]
        selected_phenotype_df: pd.DataFrame = phenotype_df[all_labels]

        # Get list of patients id
        patients_s: pd.Series = selected_phenotype_df['ScanDir ID']

        # Process and collect time series files
        time_series_mean_list: List[pd.Series] = list()
        time_series_patient_id_list: List[str] = list()
        for patient_id in patients_s:
            # Get patient time series
            time_series_path = create_patient_time_series_path(institute, patient_id, atlas, DataScope.TRAIN)
            if path.exists(time_series_path):
                patient_time_series_mean: pd.Series = time_series_mean(time_series_path)
                time_series_mean_list.append(patient_time_series_mean)
                time_series_patient_id_list.append(patient_id)

        # Transform time series files into DataFrame
        time_series_mean_aggregate: pd.Series = pd.concat(time_series_mean_list, axis=1)
        time_series_mean_df = time_series_mean_aggregate.transpose()
        time_series_id_column: pd.DataFrame = pd.DataFrame({'ID': time_series_patient_id_list})
        time_series_with_id: pd.DataFrame = pd.concat([time_series_id_column, time_series_mean_df], axis=1)
        # Put together time series and phenotype data
        patient_data_df: pd.DataFrame = time_series_with_id.set_index('ID').join(
            selected_phenotype_df.set_index('ScanDir ID'))

        return patient_data_df
