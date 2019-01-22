import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from icns.common import PhenotypeLabels, Institute, Atlas
from icns.data_functions import create_training_data, create_adhd_withheld_data

# Experiment parameters
institute = Institute.NYU
atlas = Atlas.AAL
is_target_binary = True

train_data: dict = create_training_data([institute], atlas,
                                        [PhenotypeLabels.SITE,
                                         PhenotypeLabels.AGE,
                                         PhenotypeLabels.GENDER,
                                         PhenotypeLabels.VERBAL_IQ,
                                         PhenotypeLabels.PERFORMANCE_IQ,
                                         PhenotypeLabels.FULL4_IQ])

# Prepare data for connectivity matrix
institute_data = train_data[institute]

time_series_list = list()
phenotypic_list = list()
adhd_labels = list()
for patient_id in institute_data.keys():
    patient_data = institute_data[patient_id]
    # Time-series
    time_series_df: pd.DataFrame = patient_data['time_series']
    time_series_matrix: np.ndarray = time_series_df.values
    time_series_list.append(time_series_matrix)
    # Target labels
    phenotypic: pd.Series = patient_data['phenotypic']
    if is_target_binary:
        phenotypic[PhenotypeLabels.DX] = 1 if phenotypic[PhenotypeLabels.DX] != 0 else phenotypic[PhenotypeLabels.DX]
    adhd_labels.append(phenotypic[PhenotypeLabels.DX])
    # phenotypic
    phenotypic_list.append(phenotypic.values)

correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
correlation_matrices = correlation_measure.fit_transform(time_series_list)

all_features = np.concatenate((correlation_matrices, phenotypic_list), axis=1)

# Compose data with phenotypic data
X_train, X_test, y_train, y_test = train_test_split(correlation_matrices,
                                                    adhd_labels,
                                                    test_size=0.33,
                                                    random_state=42)

classifier = LinearSVC(random_state=42)
classifier.fit(X_train, y_train)

y_train_predicted = classifier.predict(X_train)

y_test_predicted = classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_predicted)
test_accuracy = accuracy_score(y_test, y_test_predicted)

print(f'Train accuracy {train_accuracy}')
print(f'Test accuracy {test_accuracy}')

# Test against withheld

withheld_data: dict = create_adhd_withheld_data([institute], atlas,
                                                [PhenotypeLabels.SITE,
                                                 PhenotypeLabels.AGE,
                                                 PhenotypeLabels.GENDER,
                                                 PhenotypeLabels.VERBAL_IQ,
                                                 PhenotypeLabels.PERFORMANCE_IQ,
                                                 PhenotypeLabels.FULL4_IQ])

# Prepare data
w_institute_data = withheld_data[institute]

time_series_list = list()
phenotypic_list = list()
adhd_labels = list()
for patient_id in institute_data.keys():
    patient_data = institute_data[patient_id]
    # Time-series
    time_series_df: pd.DataFrame = patient_data['time_series']
    time_series_matrix: np.ndarray = time_series_df.values
    time_series_list.append(time_series_matrix)
    # Target labels
    phenotypic: pd.Series = patient_data['phenotypic']
    if is_target_binary:
        phenotypic[PhenotypeLabels.DX] = 1 if phenotypic[PhenotypeLabels.DX] != 0 else 0
    adhd_labels.append(phenotypic[PhenotypeLabels.DX])
    # phenotypic
    phenotypic_list.append(phenotypic.values)

correlation_matrices = correlation_measure.fit_transform(time_series_list)

all_features = np.concatenate((correlation_matrices, phenotypic_list), axis=1)

y_withheld_predicted = classifier.predict(correlation_matrices)

withheld_accuracy = accuracy_score(adhd_labels, y_withheld_predicted)

print(f'Withheld accuracy {withheld_accuracy}')
