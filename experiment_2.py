import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from icns.common import Phenotypic, Institute, Atlas, Features, Target
from icns.data_functions import create_training_data

# Experiment parameters
institute = Institute.OHSU
atlas = Atlas.AAL
features_composition = Features.TIME_SERIES
target_domain = Target.TD_ADHD

train_data: dict = create_training_data([institute], atlas,
                                        [Phenotypic.SITE,
                                         Phenotypic.AGE,
                                         Phenotypic.GENDER,
                                         Phenotypic.VERBAL_IQ,
                                         Phenotypic.PERFORMANCE_IQ,
                                         Phenotypic.FULL4_IQ])

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
    if target_domain is Target.TD_ADHD:
        phenotypic[Phenotypic.DX] = 1 if phenotypic[Phenotypic.DX] != 0 else phenotypic[Phenotypic.DX]
    adhd_labels.append(phenotypic[Phenotypic.DX])
    # phenotypic
    phenotypic_list.append(phenotypic.values)

correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
correlation_matrices = correlation_measure.fit_transform(time_series_list)

# Possibly combine features
patient_features = None
if features_composition is Features.TIME_SERIES:
    patient_features = correlation_matrices
elif features_composition is Features.TIME_SERIES_AND_PHENOTYPIC:
    patient_features = np.concatenate((correlation_matrices, phenotypic_list), axis=1)

# Scale features
scaled_patient_features = StandardScaler().fit_transform(patient_features)

# Compose data with phenotypic data
X_train, X_test, y_train, y_test = train_test_split(scaled_patient_features,
                                                    adhd_labels,
                                                    test_size=0.33,
                                                    random_state=42)

svc = LinearSVC(random_state=42)
rf = RandomForestClassifier(random_state=42, n_estimators=200)
sgd = SGDClassifier(random_state=42)

classifier = rf
classifier.fit(X_train, y_train)

y_train_predicted = classifier.predict(X_train)
y_test_predicted = classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_predicted)
test_accuracy = accuracy_score(y_test, y_test_predicted)
precision = precision_score(y_test, y_test_predicted)
recall = recall_score(y_test, y_test_predicted)

print(f'Train accuracy {train_accuracy}')
print(f'Test accuracy {test_accuracy}')
print(f'Precision {precision}')
print(f'Recall {recall}')