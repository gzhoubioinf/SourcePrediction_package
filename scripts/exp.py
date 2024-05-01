"""
===============
4. Experiments
===============
"""


import os

import numpy as np
import pandas as pd
import xarray as xr

from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ROCAUC

from ai4water import Model
from ai4water.utils.utils import TrainTestSplit
from ai4water.experiments import MLClassificationExperiments

from SeqMetrics import ClassificationMetrics

from utils import load_data_from_pickle, plot_confidence_interval, get_data

USE_TRAINED_MODEL = False

# %%

def f1_score_macro(t,p)->float:
    if (p == p[0]).all():
        return 0.1
    return ClassificationMetrics(t, p).f1_score(average="macro")


# %%

target = 'source_prediction'

X, y, input_features = get_data('data_5_test.nc', 'inputs_5.csv')

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(X, y)

# %%

exp = MLClassificationExperiments(
    input_features=input_features,
    output_features=target)

exp.fit(X, y.values, exclude=['AdaBoostClassifier',
        'BaggingClassifier'])

# %%

exp.compare_errors('f1_score_macro', X, y.values)
