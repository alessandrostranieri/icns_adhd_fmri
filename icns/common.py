import os
from string import Template
import configparser
from enum import Enum

config_parser = configparser.ConfigParser()
config_parser.read('./config.ini')

data_root_path: str = config_parser['PATHS']['data_root']
train_data_path: str = config_parser['PATHS']['train_data']
test_data_path: str = config_parser['PATHS']['test_data']


class DataScope(Enum):
    TRAIN = 0
    TEST = 1


institutes = ['KKI',
              'NeuroIMAGE',
              'NYU',
              'OHSU',
              'Peking_1',
              'Peking_2',
              'Peking_3',
              'Pittsburgh',
              'WashU']


class PhenotypeLabels:
    SCAN_DIR_ID = 'ScanDir ID'
    DX: str = 'DX'
    VERBAL_IQ: str = 'Verbal IQ'
    PERFORMANCE_IQ: str = 'Performance IQ'
    FULL4_IQ: str = 'Full4 IQ'


# Scanner Files
t1_file_format = Template('wssd${subject}_session_${session}_anat.nii.gz')
gm_file_format = Template('wssd${subject}_session_${session}_anat_gm.nii.gz')
rsfrmi_file_format = Template('snwmrda${subject}_session_${session}_rest_${scan}.nii.gz')
filtered_rsfrmi_file_format = Template('sfnwmrda${subject}_session_${session}_rest_${scan}.nii.gz')
time_series_file_format = Template('snwmrda${subject}_session_1_rest_1_${atlas}_TCs.1D')
smoothed_time_series_file_format = Template('sfnwmrda${subject}_session_1_rest_1_${atlas}_TCs.1D')

# Time series
train_time_series_path: str = os.path.join(train_data_path, 'TC')
test_time_series_path: str = os.path.join(test_data_path, 'TC')
time_series_templates_path: str = os.path.join(train_time_series_path, 'templates')

# Template files
atlas_types = ['cc200', 'aal']
atlas_file_map = {'cc200': 'ADHD200_parcellate_200.nii.gz', 'aal': 'aal_mask_pad.nii.gz'}


def create_template_path(atlas_name: str) -> str:
    return os.path.join(train_time_series_path, atlas_file_map[atlas_name])


def create_time_series_path(institute: str, scope: DataScope) -> str:
    if scope == DataScope.TRAIN:
        return os.path.join(train_time_series_path, institute)
    elif scope == DataScope.TEST:
        return os.path.join(test_time_series_path, institute)


def create_phenotype_path(institute: str, scope: DataScope):
    phenotype_file_name: str = f'{institute}_phenotypic.csv'
    return os.path.join(create_time_series_path(institute, scope), phenotype_file_name)


def create_patient_time_series_path(institute: str, patient_id: str, atlas: str, scope: DataScope, smoothed: bool=False) -> str:
    time_series_path = create_time_series_path(institute, scope)
    if smoothed:
        time_series_file_name = smoothed_time_series_file_format.substitute(subject=patient_id, atlas=atlas)
    else:
        time_series_file_name = time_series_file_format.substitute(subject=patient_id, atlas=atlas)
    return os.path.join(time_series_path, patient_id, time_series_file_name)
