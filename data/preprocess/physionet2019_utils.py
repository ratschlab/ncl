import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import clip_dataset, put_static_first


def forward_impute(values):
    """Fowards imputes the raw inputs.

    Contrary to MIMIC-III, no need to change resolution as time-series are already regularly samples.
    Args:
        values: Sparse time-series input

    Returns:
        Imputed time-series
    """
    values_c = np.copy(values)
    previous_row = values_c[0]
    imputed_array = [previous_row]
    for k in range(1, values_c.shape[0]):
        new_row = values_c[k]
        for i, v in enumerate(values_c[k]):
            if np.isnan(v):
                new_row[i] = previous_row[i]
        imputed_array.append(new_row)
        previous_row = imputed_array[-1]
    imputed_array = np.stack(imputed_array)
    return imputed_array


def extract_raw_data(input_files, columns, splits):
    """Function transforming the data provided by Physionet 2019 challenge to the h5 format we desire.

    Args:
        input_files: List of path to input files from  Physionet 2019 challenge.
        columns: List of columns to keep from the initial dataset.
        splits: Dict containing the split to which belong each patient.

    Returns:
        lookup_table: dict with each split as a big array.
        labels: dict with each split and and labels array in same order as lookup_table.
        patient_start_stop_ids: dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
        tasks: List of columns name in labels.
    """

    lookup_table = {'train': [], 'test': [], 'val': []}
    patient_start_stop_ids = {'train': [], 'test': [], 'val': []}
    labels = {'train': [], 'test': [], 'val': []}
    missing_pid = 0
    for patient_path in tqdm(input_files):
        df_inputs = pd.read_csv(patient_path, sep='|')
        patient_info = forward_impute(df_inputs[columns].values)
        patient_sepsis_labels = df_inputs['SepsisLabel'].values
        patient_labels = np.reshape(patient_sepsis_labels, (-1, 1))
        pid = int(patient_path.split('/')[-1].strip('p').strip('.psv'))
        if int(pid) in splits['train']:
            split = 'train'
        elif int(pid) in splits['test']:
            split = 'test'
        elif int(pid) in splits['val']:
            split = 'val'
        else:  # Removed samples by SeFt paper, we do the same for fair comparison
            split = None
            missing_pid += 1
        if split:
            if patient_start_stop_ids[split]:
                start = patient_start_stop_ids[split][-1][1] + 1
            else:
                start = 0
            stop = start + len(patient_info) - 1
            patient_start_stop_ids[split] += [[start, stop, pid]]
            lookup_table[split].append(patient_info)
            labels[split].append(patient_labels)

    lookup_table['train'] = np.concatenate(lookup_table['train'], axis=0)
    lookup_table['test'] = np.concatenate(lookup_table['test'], axis=0)
    lookup_table['val'] = np.concatenate(lookup_table['val'], axis=0)
    labels['train'] = np.concatenate(labels['train'], axis=0)
    labels['test'] = np.concatenate(labels['test'], axis=0)
    labels['val'] = np.concatenate(labels['val'], axis=0)
    patient_start_stop_ids['train'] = np.array(patient_start_stop_ids['train'])
    patient_start_stop_ids['test'] = np.array(patient_start_stop_ids['test'])
    patient_start_stop_ids['val'] = np.array(patient_start_stop_ids['val'])
    tasks = ['sepsis']

    return lookup_table, labels, patient_start_stop_ids, tasks


def physionet_to_h5(base_path, splits, var_range, static_col=['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']):
    """Wrapper around full pre-process pipeline for physionet2019.

    Args:
        base_path: Path to source data file containing the 40336 .psv files for each stay.
        splits: Dict matching stay to split.
        var_range: Dict with range for variable with false entries.
        static_col: List with name of the static columns.

    Returns:

    """
    input_files = [os.path.join(base_path, k) for k in os.listdir(base_path)]
    df_test = pd.read_csv(input_files[0], sep='|')
    columns = np.array(df_test.columns)
    # We remove sepsis labels of input columns
    columns = np.delete(columns, np.where(columns == 'SepsisLabel'))
    # We reorder columns to have Hours first
    hours_idx = list(np.where(columns == 'ICULOS')[0])
    reorder_idx = hours_idx + [k for k in range(len(columns)) if k not in hours_idx]
    columns = columns[reorder_idx]
    data_d, labels_d, patient_window_d, tasks = extract_raw_data(input_files, columns, splits)

    # We rename it Hours for consistency
    columns[0] = 'Hours'

    clipped_data = clip_dataset(var_range, data_d, columns)
    data_inverted, col_inverted = put_static_first(clipped_data, columns, static_col)
    return data_inverted, labels_d, patient_window_d, col_inverted, tasks
