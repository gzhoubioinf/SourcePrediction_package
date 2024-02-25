"""
==========================
2. Prediction Performance
==========================
"""

import numpy as np

import matplotlib.pyplot as plt

from utils import load_data_from_pickle, plot_confidence_interval


# %%

flag = ['precision', 'recall', 'f1-score']
# Define the correspondence between cut-offs and suffix numbers

filename = f'HAHHresult_kmer_cutoff0.01_file1.pkl'

datasets = load_data_from_pickle(filename)
class_label = datasets['class_label']

tmp = []  # Temporary list to store labels with values <= 1

# Extracting keys (labels) from datalabel where the value is <= 1
for k in class_label:
    if class_label.get(k) <= 1:
        tmp.append(k)
key = tmp.copy()
# Re-arranging keys in 'key' dictionary based on values in datalabel
for i in range(len(key)):
    key[class_label[tmp[i]]] = tmp[i]
# for cut_off, suffix in cut_off_to_suffix.items():
#     filename = f'HHAAresult_kmer_cutoff{cut_off}_file-1_2023{suffix}.pkl'
#     datasets[f'{cut_off}_{suffix}'] = load_data_from_pickle(filename)
#     class_label = load_data_from_pickle(filename)['class_label']

# fig, ax = plt.subplots()
width = 0.35
bars_per_dataset = len(flag) * len(key)
group_width = len(flag) * width

report = datasets['kfold_dataset']['report']
report_dataset = plot_confidence_interval(class_label, report)

report_ave = report_dataset['report_ave']
report_low = report_dataset['report_low']
report_up = report_dataset['report_up']

# %%

x = np.arange(2)
width = 0.2


fig, ax = plt.subplots()
flag = ['precision', 'recall', 'f1-score']
for i, fg in enumerate(flag):
    vals = [report_ave[name][0][fg][0] for name in key]
    errs = [
        (report_ave[name][0][fg][0] - report_low[name][0][fg][0],
         report_up[name][0][fg][0] - report_ave[name][0][fg][0])  for name in key]
    ax.bar(x + i * width, vals, width, label=fg, yerr=np.transpose(errs))
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics (XGBoost_kmer) File =2000')
ax.set_xticks(x + width)
ax.set_xticklabels(key)
ax.set_ylim([0.5,1])
ax.legend()
plt.savefig('f2000all.png', dpi=300)
plt.show()