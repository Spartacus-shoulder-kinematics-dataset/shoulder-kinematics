import numpy as np
import pandas as pd
from spartacus import DatasetCSV


def test_no_nan_in_columns():
    # Doesnt work yet online
    pass

    file = DatasetCSV.CLEAN.value
    df = pd.read_csv(file)
