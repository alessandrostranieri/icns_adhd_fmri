import logging

import pandas as pd
import matplotlib.pyplot as plt

from icns import common
from icns.common import DataScope

institute_diagnosis = dict()
for institute in common.Institute.get_directories():
    phenotype_file_path: str = common.create_phenotype_path(institute, DataScope.TRAIN)
    logging.debug(f'Reading phenotype file {phenotype_file_path}')

    phenotype_df: pd.DataFrame = pd.read_csv(phenotype_file_path)

    diagnosis_counts: pd.Series = phenotype_df.DX.value_counts()
    diagnosis_counts.sort_index(inplace=True)
    institute_diagnosis[institute] = diagnosis_counts

fig, ax = plt.subplots(figsize=(12, 8))
fig: plt.Figure = fig
ax: plt.Axes = ax
width = 1.0
diagnosis_colors = {0: 'green',
                    1: 'red',
                    2: 'orange',
                    3: 'yellow'}
diagnosis_labels = {0: 'Typically Developing Children',
                    1: 'ADHD-Combined',
                    2: 'ADHD-Hyperactive/Impulsive',
                    3: 'ADHD-Inattentive'}
bar_positions = list()
legend_handles = [None] * 4
for pos, institute in enumerate(institute_diagnosis.keys()):
    single_institute_diagnosis = institute_diagnosis[institute]
    bar_position = int(pos * 2 * width)
    bar_positions.append(bar_position)
    # TD
    bottom = 0
    for (dx_index, dx_bin) in single_institute_diagnosis.iteritems():
        legend_handles[dx_index] = ax.bar(bar_position, dx_bin, width=width, bottom=bottom,
                                          color=diagnosis_colors[dx_index],
                                          label=diagnosis_labels[dx_index])
        bottom += dx_bin

ax.set_ylabel('Diagnosis sample size')
ax.set_xlabel('Institutes')
ax.set_title('Diagnosis size by institute')
ax.set_xticks(bar_positions)
ax.set_xticklabels(common.Institute.get_directories(), rotation=45)

plt.legend(legend_handles, ('Typically Developing Children',
                            'ADHD-Combined',
                            'ADHD-Hyperactive/Impulsive',
                            'ADHD-Inattentive'))
plt.show()
