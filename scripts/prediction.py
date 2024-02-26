"""
==========================
2. Prediction Performance
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
    model={'XGBClassifier': {'n_estimators': 33,
                              'boosting_type': 'goss',
                              'num_leaves': 479,
                              'learning_rate': 0.07285550882062408}},
    input_features=list(inputs),
    output_features=target,
    verbosity=0,
)

model.reset_global_seed(313)

model.fit(x=TrainX, y=TrainY.values)

test_p = model.predict(TestX)

# %%

print(f'F1 score: {f1_score_macro(TestY.values, test_p)}')


# %%

roc_func = RocCurveDisplay.from_estimator
pr_func = PrecisionRecallDisplay.from_estimator

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                               figsize=(9,9))

tr_kws = {'estimator': model,
           'X': TrainX,
           'y': TrainY.values,
           'ax': ax1,
           'name': target
           }

roc_func(**tr_kws)
ax1.grid(ls='--', color='lightgrey')
ax1.set_title("Training")

test_kws = {'estimator': model,
       'X': TestX,
       'y': TestY.values,
       'ax': ax2,
       'name': target
       }

roc_func(**test_kws)

ax2.grid(ls='--', color='lightgrey')
ax2.set_title("Test")
ax2.set_ylabel('')

tr_kws['ax'] = ax3
pr_func(**tr_kws)
ax3.grid(ls='--', color='lightgrey')

test_kws['ax'] = ax4
pr_func(**test_kws)
ax4.set_ylabel('')
ax4.grid(ls='--', color='lightgrey')

plt.show()

# %%

# confusion matrix

X = np.concatenate([TrainX, TestX])
y = np.concatenate([TrainY, TestY])

pred = model.predict(X)

processor = ProcessPredictions('classification',
                               show=False,
                               save=False)

im = processor.confusion_matrix(
    y, pred,
    cbar_params = {"border": False},
    annotate_kws = {'fontsize': 20, "fmt": '%.f', 'ha':"center"})
ax_ = im.axes

ax_.set_title(target)
plt.show()

# # %%
#
# flag = ['precision', 'recall', 'f1-score']
# # Define the correspondence between cut-offs and suffix numbers
#
# filename = f'HAHHresult_kmer_cutoff0.01_file1.pkl'
#
# datasets = load_data_from_pickle(filename)
# class_label = datasets['class_label']
#
# tmp = []  # Temporary list to store labels with values <= 1
#
# # Extracting keys (labels) from datalabel where the value is <= 1
# for k in class_label:
#     if class_label.get(k) <= 1:
#         tmp.append(k)
# key = tmp.copy()
# # Re-arranging keys in 'key' dictionary based on values in datalabel
# for i in range(len(key)):
#     key[class_label[tmp[i]]] = tmp[i]
# # for cut_off, suffix in cut_off_to_suffix.items():
# #     filename = f'HHAAresult_kmer_cutoff{cut_off}_file-1_2023{suffix}.pkl'
# #     datasets[f'{cut_off}_{suffix}'] = load_data_from_pickle(filename)
# #     class_label = load_data_from_pickle(filename)['class_label']
#
# # fig, ax = plt.subplots()
# width = 0.35
# bars_per_dataset = len(flag) * len(key)
# group_width = len(flag) * width
#
# report = datasets['kfold_dataset']['report']
# report_dataset = plot_confidence_interval(class_label, report)
#
# report_ave = report_dataset['report_ave']
# report_low = report_dataset['report_low']
# report_up = report_dataset['report_up']
#
# # %%
#
# x = np.arange(2)
# width = 0.2
#
#
# fig, ax = plt.subplots()
# flag = ['precision', 'recall', 'f1-score']
# for i, fg in enumerate(flag):
#     vals = [report_ave[name][0][fg][0] for name in key]
#     errs = [
#         (report_ave[name][0][fg][0] - report_low[name][0][fg][0],
#          report_up[name][0][fg][0] - report_ave[name][0][fg][0])  for name in key]
#     ax.bar(x + i * width, vals, width, label=fg, yerr=np.transpose(errs))
# ax.set_ylabel('Scores')
# ax.set_title('Performance Metrics (XGBoost_kmer) File =2000')
# ax.set_xticks(x + width)
# ax.set_xticklabels(key)
# ax.set_ylim([0.5,1])
# ax.legend()
# plt.savefig('f2000all.png', dpi=300)
# plt.show()