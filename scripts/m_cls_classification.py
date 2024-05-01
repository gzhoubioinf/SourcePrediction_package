"""
==========================
3. Prediction Performance
==========================
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
from ai4water.postprocessing import ProcessPredictions

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

model_name = 'LogisticRegression'

path = '/ibex/user/iftis0a/source_prediction/results/20240424_004545'

if USE_TRAINED_MODEL:
    cpath = os.path.join(path, 'config.json')
    model = Model.from_config_file(config_path=cpath)
    wpath = os.path.join(path,'weights', 'XGBClassifier')
    model.update_weights(wpath)
else:
    model = Model(
        model=model_name,
        input_features=input_features,
        output_features=target,
        verbosity=0,
        cross_validator={"KFold": {"n_splits": 10}},
                )

    model.reset_global_seed(313)

    model = model.fit(x=TrainX, y=TrainY.iloc[:,0].astype(int))

# # %%
#
# # Get feature importance scores
# feature_importance = model.feature_importances_
#
# # Create a DataFrame with feature names and importance scores
# feature_importance_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_importance})
#
# # Sort the DataFrame by importance scores in descending order
# sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
#
# # Write the sorted DataFrame to a CSV file
# sorted_feature_importance_df.to_csv('feature_importance_05.csv', index=False)

# %%

train_p = model.predict(TrainX)

test_p = model.predict(TestX)

# %%

metrics = ClassificationMetrics(TestY.values, test_p)
print(f'Accuracy: {metrics.accuracy()}')

# %%

print(f'Precision: {metrics.precision()}')

# %%

print(f'Recall: {metrics.recall()}')

# %%

print(f'F1 score: {f1_score_macro(TestY.values, test_p)}')

# %%

print(model.predict_proba(TestX))


# %%

processor = ProcessPredictions('classification',
                               show=False,
                               save=False)

# %%

# Confusion matrix for Training samples

im = processor.confusion_matrix(
   TrainY.values, train_p,
    cbar_params = {"border": False},
    annotate_kws = {'fontsize': 20, "fmt": '%.f', 'ha':"center"})
ax_ = im.axes

ax_.set_title(target)
plt.savefig('figures/confusion_matrix_training', dpi=600)

# %%

# Confusion matrix for Test samples

im = processor.confusion_matrix(
   TestY.values, test_p,
    cbar_params = {"border": False},
    annotate_kws = {'fontsize': 20, "fmt": '%.f', 'ha':"center"})
ax_ = im.axes

ax_.set_title(target)
plt.savefig('figures/confusion_matrix_test', dpi=600)

# %%
plt.close('all')


classes = ["0", "1", "2"]
# Instantiate the classification model and visualizer
visualizer = ClassPredictionError(
    XGBClassifier(random_state=313, n_estimators=10), classes=classes
)

# Fit the training data to the visualizer
visualizer.fit(TrainX, TrainY)

# Evaluate the model on the test data


visualizer.score(TestX, TestY.values.reshape(-1,))

# Draw visualization
visualizer.show("figures/class_pred_error.png", dpi=600, bbox_inches="tight")

# %%
plt.close('all')

visualizer = ROCAUC(model, classes=["0", "1", "2"])

visualizer.fit(TrainX, TrainY)        # Fit the training data to the visualizer
visualizer.score(TestX, TestY.values.reshape(-1,))   # Evaluate the model on the test data
visualizer.show("figures/ROC_AUC.png", dpi=600, bbox_inches="tight")

# %%
plt.close('all')


viz = PrecisionRecallCurve(XGBClassifier(random_state=313, n_estimators=10), classes=classes)
viz.fit(TrainX, TrainY.iloc[:,0].astype(int))        # Fit the training data to the visualizer
viz.score(TestX, TestY.values.reshape(-1,))        # Evaluate the model on the test data
viz.show("figures/PrecisionRecallCurve.png", dpi=600, bbox_inches="tight")