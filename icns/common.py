import os
from string import Template
import configparser
from enum import Enum, unique
from typing import List

config_parser = configparser.ConfigParser()
config_parser.read('./config.ini')

data_root_path: str = config_parser['PATHS']['data_root']
train_data_path: str = config_parser['PATHS']['train_data']
test_data_path: str = config_parser['PATHS']['test_data']


@unique
class Features(Enum):
    TIME_SERIES = 0
    TIME_SERIES_AND_PHENOTYPIC = 1

    def __str__(self):
        if self is Features.TIME_SERIES:
            return 'ts'
        elif self is Features.TIME_SERIES_AND_PHENOTYPIC:
            return 'tsph'


@unique
class Target(Enum):
    TD_ADHD = 0
    TD_SUB_ADHD = 1


@unique
class DataScope(Enum):
    TRAIN = 0
    TEST = 1


@unique
class Institute(Enum):
    PEKING = 1
    BROWN = 2
    KKI = 3
    NEURO_IMAGE = 4
    NYU = 5
    OHSU = 6
    PITTSBURGH = 7
    WASHINGTON = 8

    def __str__(self):
        if self == Institute.PEKING:
            return 'Peking'
        elif self == Institute.KKI:
            return 'KKI'
        elif self == Institute.NEURO_IMAGE:
            return 'NeuroIMAGE'
        elif self == Institute.NYU:
            return 'NYU'
        elif self == Institute.OHSU:
            return 'OHSU'
        elif self == Institute.WASHINGTON:
            return 'WashU'
        else:
            return ''

    def get_code(self):
        return self.value

    @staticmethod
    def get_directories() -> List[str]:
        return ['Peking', 'KKI', 'NeuroIMAGE', 'NYU', 'OHSU', 'WashU']


class Phenotypic:
    SCAN_DIR_ID: str = 'ScanDir ID'
    SITE: str = 'Site'
    ADHD_MEASURE: str = 'ADHD Measure'
    IQ_MEASURE: str = 'IQ Measure'
    AGE: str = 'Age'
    GENDER: str = 'Gender'
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

all_phenotypic_results_file_name: str = 'allSubs_testSet_phenotypic_dx.csv'
all_phenotypic_results_path: str = os.path.join(test_data_path, all_phenotypic_results_file_name)


# Template files

class Atlas(Enum):

    CC200 = 'cc200'
    AAL = 'aal'

    def __str__(self):
        return f'{self.value}'

    def file_name(self) -> str:
        return atlas_file_map[self]


atlas_file_map = {Atlas.CC200: 'ADHD200_parcellate_200.nii.gz', Atlas.AAL: 'aal_mask_pad.nii.gz'}


def create_time_series_path(institute: str, scope: DataScope) -> str:
    if scope == DataScope.TRAIN:
        return os.path.join(train_time_series_path, institute)
    elif scope == DataScope.TEST:
        return os.path.join(test_time_series_path, institute)


def create_phenotype_path(institute: str, scope: DataScope):
    phenotype_file_name: str = f'{institute}_phenotypic.csv'
    return os.path.join(create_time_series_path(institute, scope), phenotype_file_name)


def create_patient_time_series_path(institute: str, patient_id: str, atlas: Atlas, scope: DataScope,
                                    smoothed: bool = False) -> str:
    time_series_path = create_time_series_path(institute, scope)
    if smoothed:
        time_series_file_name = smoothed_time_series_file_format.substitute(subject=patient_id, atlas=str(atlas))
    else:
        time_series_file_name = time_series_file_format.substitute(subject=patient_id, atlas=str(atlas))
    return os.path.join(time_series_path, patient_id, time_series_file_name)


def create_template_path(atlas: Atlas) -> str:
    return os.path.join(time_series_templates_path, atlas.file_name())
