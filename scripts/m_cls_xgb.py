"""
==========================
3. Prediction Performance
==========================
"""

import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.model_selection import cross_val_score, KFold

from ai4water.utils.utils import TrainTestSplit
from ai4water.postprocessing import ProcessPredictions
from easy_mpl import bar_chart
from SeqMetrics import ClassificationMetrics

from utils import read_data, feature_importances

USE_TRAINED_MODEL = False

tr_per = []
tst_per = []

# %%

def f1_score_macro(t,p)->float:
    if (p == p[0]).all():
        return 0.1
    return ClassificationMetrics(t, p).f1_score(average="macro")


# %%

target = 'source_prediction'
model_name = 'XGBClassifier'

X, y, input_features = read_data('data_5_test.nc', 'inputs_5.csv')

TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(X, y)

# %%

path = '/ibex/user/iftis0a/source_prediction/results/20240424_004545'

if USE_TRAINED_MODEL:
    model = joblib.load(path)
else:
    model = XGBClassifier(verbosity=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=313)

    # Perform k-fold cross-validation
    scores = cross_val_score(model, TrainX, TrainY.iloc[:,0].astype(int), cv=kfold)

    model = model.fit(x=TrainX, y=TrainY.iloc[:,0].astype(int))

    # joblib.dump(model, path)

# %%
# Feature Importance

feature_importances(model, input_features, 'feature_importance.csv')

# %%

# SHAP
explainer = shap.Explainer(model, TrainX)

shap_values = explainer(TrainX)

shap.summary_plot([shap_values[:, :, class_ind].values for class_ind in range(shap_values.shape[-1])], TrainX.values,
                  plot_type="bar", class_names= ['HA', 'HH', 'AA'],
                  feature_names=np.arange(92912).astype(str))

# %%

# Power Analysis

def fit_predict_on_fraction_of_data(fraction_of_data):

    data_points = int(TrainX.shape[0] * fraction_of_data)

    new_train_X = TrainX.iloc[:data_points, :]
    new_train_Y = TrainY.iloc[:data_points, 0]

    model.fit(new_train_X, new_train_Y.astype(int))

    train_p = model.predict(new_train_X)
    tr_per.append(f1_score_macro(new_train_Y.values, train_p))

    test_p = model.predict(TestX)
    tst_per.append(f1_score_macro(TestY.values, test_p))

    return train_p, test_p


# 20 %

_ = fit_predict_on_fraction_of_data(0.2)

# 40 %

_ = fit_predict_on_fraction_of_data(0.4)

# 60 %

_ = fit_predict_on_fraction_of_data(0.6)

# 80 %

_ = fit_predict_on_fraction_of_data(0.8)

# 100 %

train_p, test_p = fit_predict_on_fraction_of_data(1)

# %%

# Convert lists to NumPy arrays
array1 = np.array(tr_per)
array2 = np.array(tst_per)

# Stack arrays horizontally to create a 2D array
result = np.column_stack((array1, array2))

names = ['20', '40', '60', '80', '100']
f, ax = plt.subplots(facecolor="#EFE9E6")

bar_chart(result, color=['#60AB7B', '#F9B234'],
          orient='v', labels=names, ax=ax,
          show=False)

ax.grid(axis='y', ls='dotted', color='lightgrey')

for spine in ax.spines.values():
    spine.set_edgecolor('lightgrey')
    spine.set_linestyle('dashed')
ax.set_xlabel('Train Data Size (%)')
ax.set_ylabel('F1 score')
plt.show()



# %%

metrics = ClassificationMetrics(TestY.values, test_p)
print(f'Accuracy: {metrics.accuracy()}')

# %%

print(f'Precision: {metrics.precision()}')

# %%

print(f'Recall: {metrics.recall()}')

# %%

print(f'F1 score: {f1_score_macro(TestY.values, test_p)}')

# # %%
#
# print(model.predict_proba(TestX))


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