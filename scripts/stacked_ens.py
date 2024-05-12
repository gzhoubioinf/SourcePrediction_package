"""
==========================
5. Stacked Ensambles
==========================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as sk_metrics
from sklearn.metrics import roc_auc_score
from yellowbrick.classifier import ROCAUC
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import PrecisionRecallCurve

from ai4water import Model
from easy_mpl import bar_chart
from SeqMetrics import ClassificationMetrics
from ai4water.utils.utils import TrainTestSplit
from ai4water.postprocessing import ProcessPredictions

from utils import read_data

USE_TRAINED_MODEL = False

# %%

def f1_score_macro(t,p)->float:
    if (p == p[0]).all():
        return 0.1
    return ClassificationMetrics(t, p).f1_score(average="macro")


# %%

target = 'source_prediction'

X, y, input_features = read_data('data_5_test.nc', 'inputs_5.csv')

# %%

X_new, TestX_new, Y_new, TestY_new = TrainTestSplit(seed=313).split_by_random(X, y)

print('X_new shape', X_new.shape)
print('TestX_new shape', TestX_new.shape)
print('Y_new shape', Y_new.shape)
print('TestY_new shape', TestY_new.shape)

# %%

TrainX, ValX, TrainY, ValY = TrainTestSplit(seed=313).split_by_random(X_new, Y_new)

print('TrainX shape', TrainX.shape)
print('ValX shape', ValX.shape)
print('TrainY shape', TrainY.shape)
print('ValY shape', ValY.shape)

# %%

# metrics lists

# F1
f1_score = []
# Precision
precision = []
# Accuracy
accuracy = []
# recall
recall = []
# AUC
auc = []
# ROC
roc_auc_score_list = []

# %%

basemodel_1 = 'XGBClassifier'
basemodel_2 = 'RandomForestClassifier'
basemodel_3 = 'LogisticRegression'
meta_model_ = 'LogisticRegression'

# %%

base_model_1 = Model(
    model=basemodel_1,
    input_features=input_features,
    output_features=target,
    verbosity=0,
)

base_model_1.fit(x=TrainX, y=TrainY.iloc[:,0])

bm_1_pred = base_model_1.predict(ValX)

metrics = ClassificationMetrics(ValY.values, bm_1_pred)
print(f'Base Model 1 -> Accuracy: {metrics.accuracy()}')
print(f'Base Model 1 -> Precision: {metrics.precision()}')
print(f'Base Model 1 -> Recall: {metrics.recall()}')
print(f'Base Model 1 -> F1 score: {metrics.f1_score(average="macro")}')
accuracy.append(metrics.accuracy())
precision.append(metrics.precision(average='macro'))
recall.append(metrics.recall(average='macro'))
f1_score.append(metrics.f1_score(average='macro'))

fpr, tpr, _ = sk_metrics.roc_curve(ValY.values, bm_1_pred, pos_label=2)
auc.append(sk_metrics.auc(fpr, tpr))

roc_auc_score_list.append(roc_auc_score(ValY.values, base_model_1._model.predict_proba(ValX), multi_class='ovr'))

# %%

base_model_2 = Model(
    model=basemodel_2,
    input_features=input_features,
    output_features=target,
    verbosity=0,
)

base_model_2.fit(x=TrainX, y=TrainY.iloc[:,0])

bm_2_pred = base_model_2.predict(ValX)

metrics = ClassificationMetrics(ValY.values, bm_2_pred)
print(f'Base Model 2 -> Accuracy: {metrics.accuracy()}')
print(f'Base Model 2 -> Precision: {metrics.precision()}')
print(f'Base Model 2 -> Recall: {metrics.recall()}')
print(f'Base Model 2 -> F1 score: {f1_score_macro(ValY.values, bm_2_pred)}')

accuracy.append(metrics.accuracy())
precision.append(metrics.precision(average='macro'))
recall.append(metrics.recall(average='macro'))
f1_score.append(metrics.f1_score(average='macro'))

fpr, tpr, _ = sk_metrics.roc_curve(ValY.values, bm_2_pred, pos_label=2)
auc.append(sk_metrics.auc(fpr, tpr))

roc_auc_score_list.append(roc_auc_score(ValY.values, base_model_2._model.predict_proba(ValX), multi_class='ovr'))

# %%

base_model_3 = Model(
    model=basemodel_3,
    input_features=input_features,
    output_features=target,
    verbosity=0,
)

base_model_3.fit(x=TrainX, y=TrainY.iloc[:,0])

bm_3_pred = base_model_3.predict(ValX)

metrics = ClassificationMetrics(ValY.values, bm_3_pred)
print(f'Base Model 3 -> Accuracy: {metrics.accuracy()}')
print(f'Base Model 3 -> Precision: {metrics.precision()}')
print(f'Base Model 3 -> Recall: {metrics.recall()}')
print(f'Base Model 3 -> F1 score: {f1_score_macro(ValY.values, bm_3_pred)}')

accuracy.append(metrics.accuracy())
precision.append(metrics.precision(average='macro'))
recall.append(metrics.recall(average='macro'))
f1_score.append(metrics.f1_score(average='macro'))

fpr, tpr, _ = sk_metrics.roc_curve(ValY.values, bm_3_pred, pos_label=2)
auc.append(sk_metrics.auc(fpr, tpr))

roc_auc_score_list.append(roc_auc_score(ValY.values, base_model_3._model.predict_proba(ValX), multi_class='ovr'))


# %%

# Combine the predictions of the base models into a single feature matrix
X_val_meta = np.column_stack((bm_1_pred, bm_2_pred, bm_3_pred))


meta_model = Model(
    model=meta_model_,
    output_features=target,
    verbosity=0,
)

meta_model.fit(x=X_val_meta, y=ValY.iloc[:,0])

# %%

# Make predictions on new data
bm_1_pred_new = base_model_3.predict(X_new)
bm_2_pred_new = base_model_3.predict(X_new)
bm_3_pred_new = base_model_3.predict(X_new)

# Combine the predictions of the base models into a single feature matrix
X_new_meta = np.column_stack((bm_1_pred_new, bm_2_pred_new, bm_3_pred_new))

# %%

# Make a prediction using the meta-model
meta_pred = meta_model.predict(X_new_meta)

# %%

# Results
metrics = ClassificationMetrics(Y_new.values, meta_pred)
print(f'Meta Model -> Accuracy: {metrics.accuracy()}')
print(f'Meta Model -> Precision: {metrics.precision()}')
print(f'Meta Model -> Recall: {metrics.recall()}')
print(f'Meta Model -> F1 score: {f1_score_macro(Y_new.values, meta_pred)}')

accuracy.append(metrics.accuracy())
precision.append(metrics.precision(average='macro'))
recall.append(metrics.recall(average='macro'))
f1_score.append(metrics.f1_score(average='macro'))

fpr, tpr, _ = sk_metrics.roc_curve(Y_new.values, meta_pred, pos_label=2)
auc.append(sk_metrics.auc(fpr, tpr))

roc_auc_score_list.append(roc_auc_score(Y_new.values, meta_model._model.predict_proba(X_new_meta), multi_class='ovr'))

# %%

processor = ProcessPredictions('classification',
                               show=False,
                               save=False)

# %%

# Confusion matrix for Test samples

im = processor.confusion_matrix(
   Y_new.values, meta_pred,
    cbar_params = {"border": False},
    annotate_kws = {'fontsize': 20, "fmt": '%.f', 'ha':"center"})
ax_ = im.axes

ax_.set_title(target)
plt.savefig('figures/confusion_matrix_stacked_meta_test', dpi=600)

# %%
plt.close('all')

meta_model = 'XGBoostClassifier'
classes = ["0", "1", "2"]
# Instantiate the classification model and visualizer
visualizer = ClassPredictionError(
    LogisticRegression(), classes=classes
)

# Fit the training data to the visualizer
visualizer.fit(X_val_meta, ValY.iloc[:,0])

# Evaluate the model on the test data


visualizer.score(X_new_meta, Y_new.values.reshape(-1,))

# Draw visualization
visualizer.show("figures/class_pred_error_stacked_meta_test.png", dpi=600, bbox_inches="tight")

# %%
plt.close('all')

visualizer = ROCAUC(LogisticRegression(), classes=["0", "1", "2"])

visualizer.fit(X_val_meta, ValY.iloc[:,0])        # Fit the training data to the visualizer
visualizer.score(X_new_meta, Y_new.values.reshape(-1,))   # Evaluate the model on the test data
visualizer.show("figures/ROC_AUC_stacked_meta_test.png", dpi=600, bbox_inches="tight")

# %%
plt.close('all')

viz = PrecisionRecallCurve(LogisticRegression(), classes=classes)
viz.fit(X_val_meta, ValY.iloc[:,0])        # Fit the training data to the visualizer
viz.score(X_new_meta, Y_new.values.reshape(-1,))        # Evaluate the model on the test data
viz.show("figures/PrecisionRecallCurve_stacked_meta_test.png", dpi=600, bbox_inches="tight")

# %%

names = ['XGB', 'RF', 'LR', 'Stacked']

f, ax = plt.subplots(facecolor="#EFE9E6")

bar_chart(f1_score, cmap='summer_r',
          orient='v', labels=names, ax=ax,
          show=False)

ax.grid(axis='y', ls='dotted', color='lightgrey')
ax.set_title('F1_score')

for spine in ax.spines.values():
    spine.set_edgecolor('lightgrey')
    spine.set_linestyle('dashed')

plt.show()
#
# %%

f, ax = plt.subplots(facecolor="#EFE9E6")

bar_chart(accuracy, cmap='summer_r',
          orient='v', labels=names, ax=ax,
          show=False)

ax.grid(axis='y', ls='dotted', color='lightgrey')
ax.set_title('Accuracy')

for spine in ax.spines.values():
    spine.set_edgecolor('lightgrey')
    spine.set_linestyle('dashed')

plt.show()
#
# # %%
#
f, ax = plt.subplots(facecolor="#EFE9E6")

bar_chart(precision, cmap='summer_r',
          orient='v', labels=names, ax=ax,
          show=False)

ax.grid(axis='y', ls='dotted', color='lightgrey')
ax.set_title('precision')

for spine in ax.spines.values():
    spine.set_edgecolor('lightgrey')
    spine.set_linestyle('dashed')

plt.show()
#
# # %%
#
f, ax = plt.subplots(facecolor="#EFE9E6")

bar_chart(recall, cmap='summer_r',
          orient='v', labels=names, ax=ax,
          show=False)

ax.grid(axis='y', ls='dotted', color='lightgrey')
ax.set_title('recall')

for spine in ax.spines.values():
    spine.set_edgecolor('lightgrey')
    spine.set_linestyle('dashed')

plt.show()

# %%

f, ax = plt.subplots(facecolor="#EFE9E6")

bar_chart(auc, cmap='summer_r',
          orient='v', labels=names, ax=ax,
          show=False)

ax.grid(axis='y', ls='dotted', color='lightgrey')
ax.set_title('AUC')

for spine in ax.spines.values():
    spine.set_edgecolor('lightgrey')
    spine.set_linestyle('dashed')

plt.show()

# %%

names = ['XGB', 'RF', 'LR', 'Stacked']
f, ax = plt.subplots(facecolor="#EFE9E6")

bar_chart(roc_auc_score_list, cmap='summer_r',
          orient='v', labels=names, ax=ax,
          show=False)

ax.grid(axis='y', ls='dotted', color='lightgrey')
ax.set_title('ROC AUC Score')

for spine in ax.spines.values():
    spine.set_edgecolor('lightgrey')
    spine.set_linestyle('dashed')

plt.show()
