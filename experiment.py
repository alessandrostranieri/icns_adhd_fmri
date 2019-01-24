import numpy as np
import pandas as pd
import warnings as wrn
from nilearn.connectome import ConnectivityMeasure
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

from icns.common import Phenotypic, Institute, Atlas, Features, Target
from icns.data_functions import create_training_data
from icns.plotting import plot_institute_scores


# Helper functions
def create_classifier(classifier_name):
    if classifier_name is 'LinearSVC':
        return LinearSVC(random_state=42)
    elif classifier_name is 'SVC':
        return SVC(random_state=42)
    elif classifier_name is 'RandomForest':
        return RandomForestClassifier(random_state=42, n_estimators=200)
    elif classifier_name is 'SGD':
        return SGDClassifier(random_state=42)


# Mute nipy warning
# wrn.simplefilter('error')

# Experiment parameters
institutes = [Institute.PEKING, Institute.NYU, Institute.OHSU]
atlas = Atlas.AAL
features_composition = Features.TIME_SERIES
phenotypic_features = list() if features_composition is Features.TIME_SERIES else [Phenotypic.SITE,
                                                                                   Phenotypic.AGE,
                                                                                   Phenotypic.GENDER,
                                                                                   Phenotypic.VERBAL_IQ,
                                                                                   Phenotypic.PERFORMANCE_IQ,
                                                                                   Phenotypic.FULL4_IQ]
target_domain = Target.TD_ADHD
connectivity_kind = 'correlation'
pca_components_number = 4
classifier_type = 'RandomForest'

# Experiment
train_data: dict = create_training_data(institutes, atlas,
                                        phenotypic_features, smoothed=True)

institute_scores = dict()
for institute in train_data.keys():

    institute_data: dict = train_data[institute]
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

    # Image features are a vectorization of the correlation from time series
    correlation_measure = ConnectivityMeasure(kind=connectivity_kind, vectorize=True)
    connectivity_vector = correlation_measure.fit_transform(time_series_list)

    # Perform PCA
    pca_transformed = connectivity_vector
    if pca_components_number:
        pca = PCA(n_components=pca_components_number, random_state=42)
        pca_transformed = pca.fit_transform(connectivity_vector)

    # Optionally combine features
    patient_features = None
    if features_composition is Features.TIME_SERIES:
        patient_features = pca_transformed
    elif features_composition is Features.TIME_SERIES_AND_PHENOTYPIC:
        patient_features = np.concatenate((pca_transformed, phenotypic_list), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(patient_features,
                                                        adhd_labels,
                                                        test_size=0.33,
                                                        random_state=42)

    # Train
    classifier = create_classifier(classifier_type)
    classifier.fit(X_train, y_train)

    # Predict
    y_train_predicted = classifier.predict(X_train)
    y_majority = [np.argmax(np.bincount(y_train))] * len(y_test)
    y_test_predicted = classifier.predict(X_test)

    # Collect results
    train_accuracy = accuracy_score(y_train, y_train_predicted)
    chance_accuracy = accuracy_score(y_test, y_majority)
    test_accuracy = accuracy_score(y_test, y_test_predicted)
    precision = precision_score(y_test, y_test_predicted)
    recall = recall_score(y_test, y_test_predicted)

    print(f'Institute: {str(institute)}')
    print(f'Train accuracy {train_accuracy}')
    print(f'Chance accuracy {chance_accuracy}')
    print(f'Test accuracy {test_accuracy}')
    print(f'Precision {precision}')
    print(f'Recall {recall}')

    institute_scores[str(institute)] = {'accuracy': test_accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'chance': chance_accuracy}

# Store results
plot_institute_scores(institute_scores,
                      filename=f'score_{classifier_type}_{connectivity_kind}_{str(features_composition)}.png'
                      .replace(' ', '-'))
