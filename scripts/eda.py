"""
=============================
1. Exploratory Data Analysis
=============================
"""


# importing necessary modules

import io
import os

import numpy as np

import matplotlib.pyplot as plt

from dython.nominal import associations

from easy_mpl.utils import create_subplots
from easy_mpl import pie

from utils import get_data

# %%



# # reading data
X, y = get_data()

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

# %%

# converting sparse matrix to a numpy array
X_array = X.toarray()

X_array.shape

# %%

y.shape

# %%

# plotting heatmap

whole_data = np.concatenate((X_array, y.values.reshape(-1, 1)), axis=1)
_ = associations(
    whole_data,
    nom_nom_assoc="cramer",
    fmt=".1f",
    figsize=(8, 8),
    plot=False,
    cbar=True
)
plt.tight_layout()
plt.show()

# %%

# plotting input features

fig, axes = create_subplots(10, figsize=(14,10))

for col, ax in zip(range(X.shape[1]), axes.flatten()):

    pie(X_array[:, col], ax=ax,
        ax_kws=dict(title=f'input feature {col+1}'), show=False)

plt.tight_layout()
plt.show()

# %%

# pie chart of target

pie(y, ax_kws=dict(title='Target'))