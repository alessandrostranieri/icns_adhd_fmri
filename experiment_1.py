import pandas as pd

from icns.common import PhenotypeLabels
from icns.data_functions import create_training_data


if __name__ == '__main__':
    # Get institute data
    train_data: pd.DataFrame = create_training_data(['NYU'], 'aal',
                                                    [PhenotypeLabels.VERBAL_IQ,
                                                     PhenotypeLabels.PERFORMANCE_IQ,
                                                     PhenotypeLabels.FULL4_IQ])
    print(f'Shape of the data-set: {train_data.shape}')
