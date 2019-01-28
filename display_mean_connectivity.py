from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure

from icns.common import Institute, Atlas, Phenotypic
from icns.data_functions import create_training_data


def plot_matrices(matrices, matrix_kind, dx: str):
    n_matrices = len(matrices)
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        # Set diagonal to zero, for better visualization
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{0}, subject {1}'.format(matrix_kind, dx)
        plotting.plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                             title=title, figure=fig, colorbar=False)


connectivity_kind = 'partial correlation'
institute = Institute.PEKING

td_time_series: List[pd.DataFrame] = list()
adhd_time_series: List[pd.DataFrame] = list()

time_series: dict = create_training_data(institute,
                                         Atlas.AAL,
                                         list(),
                                         smoothed=True)

for patient_id in time_series.keys():

    patient_data = time_series[patient_id]

    diagnosis = patient_data['phenotypic'][Phenotypic.DX]
    patient_data_values = patient_data['time_series'].values

    if diagnosis == 0:
        td_time_series.append(patient_data_values)
    else:
        adhd_time_series.append(patient_data_values)

dx = 'TD'
connectivity_measure = ConnectivityMeasure(kind=connectivity_kind)
if dx == 'TD':
    partial_correlation_matrices = connectivity_measure.fit_transform(td_time_series)
else:
    partial_correlation_matrices = connectivity_measure.fit_transform(adhd_time_series)

plot_matrices(partial_correlation_matrices[:4], connectivity_kind, dx=dx)

plt.show()