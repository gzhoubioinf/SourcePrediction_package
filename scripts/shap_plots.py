"""
========
4. SHAP
========
"""

import shap
import pickle
import matplotlib.pyplot as plt

# %%

filename = f'HAHHresult_kmer_cutoff0.01_file1.pkl'

with open(filename, 'rb') as f:
    re = pickle.load(f)
class_label = re['class_label']
tmp = []  # Temporary list to store labels with values <= 1

# %%

# Extracting keys (labels) from datalabel where the value is <= 1
for k in class_label:
    if class_label.get(k) <= 1:
        tmp.append(k)
key = tmp.copy()
# Re-arranging keys in 'key' dictionary based on values in datalabel
for i in range(len(key)):
    key[class_label[tmp[i]]] = tmp[i]
shape_value_dataset = re['shape_value_dataset']
shap_values = shape_value_dataset['shap_values']
exp_set = shape_value_dataset['exp_set']
sub_bestfeature_name = shape_value_dataset['sub_bestfeature_name']
sub_bestfeature_indices = shape_value_dataset['sub_bestfeature_indices']

# %%

shap.summary_plot(shap_values, exp_set.toarray(),
                  feature_names=sub_bestfeature_name,
                  class_names=key,
                  # plot_type='violin',
                  show=False)
# Set the y-tick labels as numbers
num_ticks = len(sub_bestfeature_indices)
# plt.yticks( range(num_ticks))

plt.yticks(range(num_ticks), range(num_ticks))

# Print correspondence between numbers and feature names
for i, name in enumerate(sub_bestfeature_name[0:min(10, len(sub_bestfeature_name))]):
    print(f"{i}: {name}")
plt.tight_layout()
plt.show()