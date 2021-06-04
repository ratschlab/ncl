import argparse
import os
import pickle

from mimic3_utils import benchmark_to_h5
from physionet2019_utils import physionet_to_h5
from sklearn.preprocessing import StandardScaler
from utils import save_to_h5_with_tasks, scaling_data_common


def build_mimic3_h5(load_path, save_path, channel_to_id, matching_dict, var_range, static_col):
    """Wrapper to build MIMIC-III benchmark dataset in the desired format for our loader.

    Args:
        load_path: String with path to source data from https://github.com/YerevaNN/mimic3-benchmarks.
        save_path: String with path where to save the final h5 file.
        channel_to_id: Dict obtained from mimic_resources/ that matches variables to id.
        matching_dict: Dict obtained from mimic_resources/ that matches string to categories.
        var_range: Dict obtained from mimic_resources/ with ranges for certain variables to remove false entries.
        static_col: Name of the static columns, should be only Height.

    Returns:

    """
    data, labels, windows, col, tasks = benchmark_to_h5(load_path, channel_to_id, matching_dict, var_range, static_col)
    save_file = os.path.join(save_path, 'non_scaled.h5')
    save_to_h5_with_tasks(save_file, col, tasks, data, labels, windows)


def build_physionet_h5(load_path, save_path, splits, var_range, static_col):
    """Wrapper to build Physionet2019 benchmark dataset in the desired format for our loader.

    Args:
        load_path: String with path to source data from https://physionet.org/content/challenge-2019/1.0.0/.
        save_path: String with path where to save the final h5 file.
        splits: Dict obtained from physionet2019_resources/ with the split to which each stay correspond.
        This splits are exactly the same as https://arxiv.org/abs/1909.12064.
        var_range: Dict obtained from physionet2019_resources/ with ranges for certain variables to remove false entries.
        static_col: Name of the static columns.

    Returns:

    """
    data, labels, windows, col, tasks = physionet_to_h5(load_path, splits, var_range, static_col)
    save_file = os.path.join(save_path, 'non_scaled.h5')
    save_to_h5_with_tasks(save_file, col, tasks, data, labels, windows)


def scale_dataset(non_scaled_path, save_path, static_col):
    """Scales dataset with StandardScaling on the non-categorical variables.

    Args:
        non_scaled_path: String with path to the non scaled dataset h5 file.
        save_path: String with path where to save the final h5 file.
        static_col: Name of the static columns.

    Returns:

    """
    static_idx = [k for k in range(len(static_col))]
    data, labels, windows, col, tasks = scaling_data_common(non_scaled_path, threshold=25, scaler=StandardScaler(),
                                                            static_idx=static_idx, df_ref=None)
    save_file = os.path.join(save_path, 'scaled.h5')
    save_to_h5_with_tasks(save_file, col, tasks, data, labels, windows)


if __name__ == "__main__":

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Name of the dataset to build, either 'mimic3' or 'physionet2019' ")
    parser.add_argument("--load_path", help="Path to folder containing MIMIC-III Benchmark source path")
    parser.add_argument("--save_path", help="Path to folder where we save the extracted dataset")
    parser.add_argument("--resource_path", help="Path to folder with the resources pickles files ")
    parser.add_argument("--static_columns", nargs='+', help="List of static columns names ")
    parser.add_argument("--scale", default=True, type=boolean_string, help="Whether or not to save a scaled copy")
    parser.add_argument("--overwrite", default=False, type=boolean_string, help="Whether to overwrite existing files")

    arg = parser.parse_args()

    if arg.dataset == 'mimic3':
        with open(os.path.join(arg.resource_path, 'channel_to_id_m3.pkl'), 'rb') as f:
            channel_to_id = pickle.load(f)
        with open(os.path.join(arg.resource_path, 'matching_dict_m3.pkl'), 'rb') as f:
            matching_dict = pickle.load(f)
        with open(os.path.join(arg.resource_path, 'var_range_m3.pkl'), 'rb') as f:
            var_range = pickle.load(f)

        save_file = os.path.join(arg.save_path, 'non_scaled.h5')
        if (not os.path.isfile(save_file)) or arg.overwrite:
            build_mimic3_h5(arg.load_path, arg.save_path, channel_to_id, matching_dict, var_range, arg.static_columns)


    elif arg.dataset == 'physionet2019':
        with open(os.path.join(arg.resource_path, 'splits_from_seft_physionet.pkl'), 'rb') as f:
            splits = pickle.load(f)
        with open(os.path.join(arg.resource_path, 'var_range_physionet.pkl'), 'rb') as f:
            var_range = pickle.load(f)

        save_file = os.path.join(arg.save_path, 'non_scaled.h5')
        if (not os.path.isfile(save_file)) or arg.overwrite:
            build_physionet_h5(arg.load_path, arg.save_path, splits, var_range, arg.static_columns)

    else:
        raise Exception("arg.dataset has to be either 'mimic3' or 'physionet2019' ")

    scale_file = os.path.join(arg.save_path, 'scaled.h5')
    if not os.path.isfile(scale_file) and arg.scale:
        scale_dataset(save_file, arg.save_path, arg.static_columns)
