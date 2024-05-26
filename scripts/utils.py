"""
==========
utils
==========
"""

# importing necessary modules

import json
import numpy as np
import scipy.stats
import pandas as pd
import xarray as xr
import netCDF4 as nc
import seaborn as sns
import matplotlib.pyplot as plt

from SeqMetrics import ClassificationMetrics

import kmer_ml

# %%

# data preprocessing
def get_data(return_bf_only=False):

    best_feature_indices = [20178, 20177, 20176, 20175, 20174, 20173, 20172, 20171, 20170,
       20169]

    with open('input.json', 'r') as f:
    #with open(os.path.join(os.path.dirname(os.getcwd()), 'scripts', 'input.json'), 'r') as f:
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

    y = filtered_df['data_type']

    if return_bf_only:
        try:
            return X[:, best_feature_indices], y
        except IndexError:
            raise IndexError(f"shape of input: {X.shape}, "
                             f"number of selected files: {numb_files_select}, "
                             f"{cutoff}, {chunkdata_path}, {removed_percent}")

    return X, y, voc_col.keys()


def save_data_to_nc(fname):

    X, y, inputs = get_data()

    X = X.toarray()

    data = np.concatenate((X, y.to_numpy().reshape(-1,1)), axis=1)

    # Create a netCDF file
    with nc.Dataset(fname, 'w') as ds:
        # Create dimensions
        ds.createDimension('samples', data.shape[0])
        ds.createDimension('inputs', data.shape[1] - 1)

        # Create a variable
        X = ds.createVariable('X', 'uint8', ('samples', 'inputs'))

        # Create a variable
        target = ds.createVariable('target', 'uint8', ('samples',))

        # Create a variable for input features
        str_var = ds.createVariable('inputs', str, ('inputs',))

        # Store data in the variable
        X[:] = data[:, 0:-1]
        target[:] = data[:, -1]
        str_var[:] = np.array(list(inputs))


def read_data(fname):

    ds = xr.open_dataset(fname)
    X = ds.X.to_pandas()
    y = ds.target.to_pandas()

    return X, y, list(X.columns)


def calculate_ci(fname):

    df = pd.read_csv(fname)
    # Number of bootstrap samples
    num_bootstrap_samples = 100

    confidence_interval = []

    for index, row in df.iterrows():
        # Bootstrap resampling
        bootstrap_importance_means = []
        for _ in range(num_bootstrap_samples):
            bootstrap_sample = np.random.choice(row, size=len(row), replace=True)
            bootstrap_importance_means.append(np.mean(bootstrap_sample))

        # Compute the confidence interval
        confidence_interval.append(scipy.stats.norm.interval(0.95, loc=np.mean(bootstrap_importance_means),
                                                             scale=np.std(bootstrap_importance_means)))

    ci = pd.DataFrame(confidence_interval, columns=['CI_lower', 'CI_upper'])

    ex_file = pd.read_csv(fname)

    ex_file['CI_lower'] = ci['CI_lower']
    ex_file['CI_upper'] = ci['CI_upper']

    ex_file.to_csv(fname, index=False, header=False)

    return


def plot_ci(fname):

    df = pd.read_csv(fname, index_col=0)

    x = np.linspace(1, 50, 50).astype(int)

    fig, ax = plt.subplots()
    ax.plot(x, df['Avg Importance'].values, '-.')
    ax.fill_between(x, df['CI_lower'].values, df['CI_upper'].values, color='b', alpha=.1)

    # Set x and y axis titles
    ax.set_xlabel('Features')
    ax.set_ylabel('Average Importance')
    plt.tight_layout()
    plt.show()

    return


def feature_importances(model, input_features, fname):
    # Get feature importance scores
    feature_importance = model.feature_importances_

    # Create a DataFrame with feature names and importance scores
    feature_importance_df = pd.DataFrame({'Feature': input_features, 'Importance': feature_importance})

    # Sort the DataFrame by importance scores in descending order
    sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Write the sorted DataFrame to a CSV file
    sorted_feature_importance_df[0:50].to_csv(fname, index=False)

    return



# %%

def box_violin(ax, data, palette=None):
    if palette is None:
        palette = sns.cubehelix_palette(start=.5, rot=-.5, dark=0.3, light=0.7)
    ax = sns.violinplot(orient='h', data=data,
                        palette=palette,
                        scale="width", inner=None,
                        ax=ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=ax.transData))

    sns.boxplot(orient='h', data=data, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
    old_len_collections = len(ax.collections)

    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return