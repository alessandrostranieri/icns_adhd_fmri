import os
from string import Template

data_root_path: str = 'C:\\Users\\deral\\University\\ICNS\\Project\\Data'

institutes = ['KKI',
              'NeuroIMAGE',
              'NYU',
              'OHSU',
              'Peking_1',
              'Peking_2',
              'Peking_3',
              'Pittsburgh',
              'WashU']

# Scanner Files
t1_file_format = Template('wssd${subject}_session_${session}_anat.nii.gz')
gm_file_format = Template('wssd${subject}_session_${session}_anat_gm.nii.gz')
rsfrmi_file_format = Template('snwmrda${subject}_session_${session}_rest_${scan}.nii.gz')
filtered_rsfrmi_file_format = Template('sfnwmrda${subject}_session_${session}_rest_${scan}.nii.gz')
tc_file_format = Template('snwmrda${subject}_session_1_rest_1_${atlas}_TCs.1D')
filtered_tc_file_format = Template('sfnwmrda${subject}_session_1_rest_1_${atlas}_TCs.1D')

# Time courses
time_series_path: str = os.path.join(data_root_path, 'TC')
time_series_templates_path: str = os.path.join(time_series_path, 'templates')

# Template files
atlas_types = ['cc200', 'aal']
atlas_file_map = {'cc200': 'ADHD200_parcellate_200.nii.gz', 'aal': 'aal_mask_pad.nii.gz'}


def create_path_checked(path, *paths):
    result = os.path.join(path, *paths)
    assert os.path.exists(result), f'Path {result} does not exist'
    return result


def create_phenotypic_path(institute):
    phenotypic_file: str = f'{institute}_phenotypic.csv'
    return create_path_checked(time_series_path, institute, phenotypic_file)


def create_time_series_path(institute, subject, atlas) -> str:
    result = os.path.join(time_series_path, institute, subject, tc_file_format.substitute(subject=subject, atlas=atlas))
    assert os.path.exists(result), f'Path {result} does not exist'
    return result


def create_template_path(atlas_name: str) -> str:
    result = os.path.join(time_series_templates_path, atlas_file_map[atlas_name])
    assert os.path.exists(result), f'Path {result} does not exist'
    return result
