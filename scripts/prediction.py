"""
==========================
3. Prediction Performance
==========================
"""

import numpy as np

import matplotlib.pyplot as plt

from ai4water import Model
from ai4water.utils.utils import TrainTestSplit
from ai4water.postprocessing import ProcessPredictions

from SeqMetrics import ClassificationMetrics

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from utils import load_data_from_pickle, plot_confidence_interval, get_data

# %%

def f1_score_macro(t,p)->float:
    if (p == p[0]).all():
        return 0.1
    return ClassificationMetrics(t, p).f1_score(average="macro")


# %%

feature_names = ['TTTTTTGCTAGCGGAAAACGGAGATTTAAAAGAAAACAAAATATTTTTTGCGTA',
                 'TTTTTTGCTAGCTGAACGTAAAACATTAAATTTCGCTCATTATTATATTATGCT',
                 'TTTTTTGCTAGGAAGTCGGCAGCCACGCGTAGTGCGCCAGTGCCACCCGGAGTC',
                 'TTTTTTGCTATAGCAATGGTAGCAGCACCATCCGTACTTAAAAACGCACTAAAT',
                 'TTTTTTGCTATATGCGGTTTGTAAGATTGATTTTTCTTCTCAAGTTCCTTGTTC',
                 'TTTTTTGCTATCAAAATGCTACCTCTCCCTTCTTGCAATAAATGACCAAGGCAC',
                 'TTTTTTGCTATCAATCTTATTGATTTAATTTCATTGCTTAATACTAACTTAATA',
                 'TTTTTTGCTATCAATGAATTCATTAATTCCTAACTCATTAATTAGATCTGCAAT',
                 'TTTTTTGCTATCACCGAAAATAGTGCGGATCCCGCATGGTATTTAGGTTTACCC',
                 'TTTTTTGCTATCAGCCGACGCTGATCGGCGTACCCGCAAATGCTGATTTGCGTT']

target = 'source_prediction'

X, y, inputs = get_data(return_bf_only=False)

X = X.toarray()

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(X, y)

# %%

model_name = 'XGBClassifier'

model = Model(
    model={'XGBClassifier': {'n_estimators': 100,
                              'boosting_type': 'dart',
                              'num_leaves': 174,
                              'learning_rate': 0.07343729923442806}},
    input_features=list(inputs),
    output_features=target,
    verbosity=0,
)

model.reset_global_seed(313)

model.fit(x=TrainX, y=TrainY.values)

train_p = model.predict(TrainX)
test_p = model.predict(TestX)

# %%

print(f'F1 score: {f1_score_macro(TestY.values, test_p)}')

# %%

processor = ProcessPredictions('classification',
                               show=False,
                               save=False)

# %%

# Confusion matrix for Trainig samples

im = processor.confusion_matrix(
   TrainY, train_p,
    cbar_params = {"border": False},
    annotate_kws = {'fontsize': 20, "fmt": '%.f', 'ha':"center"})
ax_ = im.axes

ax_.set_title(target)
plt.show()

# %%

# Confusion matrix for Test samples

im = processor.confusion_matrix(
   TestY, test_p,
    cbar_params = {"border": False},
    annotate_kws = {'fontsize': 20, "fmt": '%.f', 'ha':"center"})
ax_ = im.axes

ax_.set_title(target)
plt.show()
