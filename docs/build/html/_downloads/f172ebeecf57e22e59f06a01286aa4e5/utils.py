"""
==========
utils
==========
"""

# importing necessary modules

import os
import json

import kmer_ml

# %%

# data preprocessing
def get_data():

    best_feature_indices = [20178, 20177, 20176, 20175, 20174, 20173, 20172, 20171, 20170,
       20169]

    with open(os.path.join(os.path.dirname(os.getcwd()), 'example', 'input.json'), 'r') as f:
        data = json.load(f)

    cutoff = data['cutoff']
    numb_files_select = data['numb_files_select']
    traits_scoary_path = data['traits_scoary_path']
    class_label = data['class_label']
    chunkdata_path = data['chunkdata_path']

    filepath  = chunkdata_path
    filtered_df = kmer_ml.get_datafilter(class_label,traits_scoary_path)
    row_list = filtered_df.transpose().columns.values

    X, voc_col, voc_row, removed_percent =kmer_ml.get_datamatrix(row_list,
                                                            numb_files_select=numb_files_select,
                                                            datapath=filepath,
                                                            cutoff=cutoff)
    # X = mtr
    y = filtered_df['data_type']

    X_subset = X[:, best_feature_indices]

    return X_subset, y