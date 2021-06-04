import os

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def put_static_first(data, col, static_col):
    """Simple function putting the static columns first in the data.

    Args:
        data: Dict with a data array for each split.
        col: Ordered list of the columns in the data.
        static_col: List of static columns names.

    Returns:
        data_inverted : Analog  to data with columns reordered in each split.
        col_inverted : Analog to col woth columns names reordered.
    """
    static_index = list(np.where(np.isin(np.array(col), static_col))[0])
    n_col = len(col)
    non_static_index = [k for k in range(n_col) if k not in static_index]
    new_idx = static_index + non_static_index
    data_inverted = {}
    for split in ['train', 'test', 'val']:
        data_inverted[split] = data[split][:, new_idx]
    col_inverted = list(np.array(col)[new_idx])
    return data_inverted, col_inverted


def clip_dataset(var_range, data, columns):
    """Set each values outside of predefined range to NaN.

    Args:
        var_range: Dict with associated range [min,max] to each variable name.
        data: Dict with a data array for each split.
        columns: Ordered list of the columns in the data.

    Returns:
        new_data : Data with no longer any value outside of the range.
    """
    new_data = {}
    for split in ['train', 'test', 'val']:
        clipped_data = data[split][:]
        for i, col in enumerate(columns):
            if var_range.get(col):
                idx = np.sort(np.concatenate([np.argwhere(clipped_data[:, i] > var_range[col][1]),
                                              np.argwhere(clipped_data[:, i] < var_range[col][0])])[:, 0])
                clipped_data[idx, i] = np.nan
        new_data[split] = clipped_data
    return new_data


def finding_cat_features(rep_data, threshold):
    """
    Extracts the index and names of categorical in a pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        threshold: Number of uniqur value below which we consider a variable as categorical if it's an integer

    Returns:
        categorical: List of names containing categorical features.
        categorical_idx: List of matching column indexes.

    """
    columns = rep_data['data'].attrs['columns']

    categorical = []

    for i, c in enumerate(columns):
        values = rep_data['data']['train'][:, i]
        values = values[~np.isnan(values)]
        nb_values = len(np.unique(values))

        if nb_values <= threshold and np.all(values == values.astype(int)):
            categorical.append(c)

    categorical_idx = np.sort([np.argwhere(columns == feat)[0, 0] for feat in categorical])

    return categorical, categorical_idx


def finding_cat_features_fom_file(rep_data, info_df):
    """
    Extracts the index and names of categorical in a pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        info_df: Dataframe with information on each variable.

    Returns:
        categorical: List of names containing categorical features.
        categorical_idx: List of matching column indexes.

    """
    columns = rep_data['data'].attrs['columns']
    categorical = []

    for i, c in enumerate(columns):
        if c.split('_')[0] != 'plain':
            pass
        else:
            if info_df[info_df['VariableID'] == c.split('_')[-1]]['Datatype'].values == 'Categorical':
                categorical.append(c)
    categorical_idx = np.sort([np.argwhere(columns == feat)[0, 0] for feat in categorical])
    return categorical, categorical_idx


def get_one_hot(rep_data, cat_names, cat_idx):
    """
    One-hots the categorical features in a given pre-built dataset.
    Args:
        rep_data: Pre-built dataset as a h5py.File(...., 'r').
        cat_names: List of names containing categorical features.
        cat_idx: List of matching column indexes.

    Returns:
        all_categorical_data: Dict with each split one-hotted categorical column as a big array.
        col_name: List of name of the matching columns

    """
    all_categorical_data = np.concatenate([rep_data['data']['train'][:, cat_idx],
                                           rep_data['data']['test'][:, cat_idx],
                                           rep_data['data']['val'][:, cat_idx]], axis=0)
    cat_dict = {}
    col_name = []
    for i, cat in enumerate(cat_idx):
        dum = np.array(pd.get_dummies(all_categorical_data[:, i]))
        if dum.shape[-1] <= 2:
            dum = dum[:, -1:]
            col_name += [cat_names[i].split('_')[-1] + '_cat']
        else:
            col_name += [cat_names[i].split('_')[-1] + '_cat_' + str(k) for k in range(dum.shape[-1])]
        cat_dict[cat] = dum

    all_categorical_data_one_h = np.concatenate(list(cat_dict.values()), axis=1)

    all_categorical_data = {}
    all_categorical_data['train'] = all_categorical_data_one_h[:rep_data['data']['train'].shape[0]]
    all_categorical_data['test'] = all_categorical_data_one_h[
                                   rep_data['data']['train'].shape[0]:rep_data['data']['train'].shape[0] +
                                                                      rep_data['data']['test'].shape[0]]
    all_categorical_data['val'] = all_categorical_data_one_h[-rep_data['data']['val'].shape[0]:]

    return all_categorical_data, col_name


def scaling_data_common(data_path, threshold=25, scaler=StandardScaler(), static_idx=None, df_ref=None):
    """
    Wrapper which one-hot and scales the a pre-built dataset.
    Args:
        data_path: String with the path to the pre-built non scaled dataset
        threshold: Int below which we consider a variable as categorical
        scaler: sklearn Scaler to use, default is StandardScaler.
        static_idx: List of indexes containing static columns.
        df_ref: Reference dataset containing supplementary information on the columns.

    Returns:
        data_dic: dict with each split as a big array.
        label_dic: dict with each split and and labels array in same order as lookup_table.
        patient_dic: dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].
        col: list of the variables names corresponding to each column.
        labels_name: list of the tasks name corresponding to labels columns.
    """
    rep_data = h5py.File(data_path, 'r')
    columns = rep_data['data'].attrs['columns']
    train_data = rep_data['data']['train'][:]
    test_data = rep_data['data']['test'][:]
    val_data = rep_data['data']['val'][:]

    # We just extract tasks name to propagate
    if 'tasks' in list(rep_data['labels'].attrs.keys()):
        labels_name = rep_data['labels'].attrs['tasks']
    else:
        labels_name = None

    # We treat np.inf and np.nan as the same
    np.place(train_data, mask=np.isinf(train_data), vals=np.nan)
    np.place(test_data, mask=np.isinf(test_data), vals=np.nan)
    np.place(val_data, mask=np.isinf(val_data), vals=np.nan)
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # We pad after scaling, Thus zero is equivalent to padding with the mean value across patient
    np.place(train_data_scaled, mask=np.isnan(train_data_scaled), vals=0.0)
    np.place(test_data_scaled, mask=np.isnan(test_data_scaled), vals=0.0)
    np.place(val_data_scaled, mask=np.isnan(val_data_scaled), vals=0.0)

    # If we have static values we take one per patient stay
    if static_idx:
        train_static_values = train_data[rep_data['patient_windows']['train'][:][:, 0]][:, static_idx]
        static_scaler = StandardScaler()
        static_scaler.fit(train_static_values)

        # Scale all entries
        train_data_static_scaled = static_scaler.transform(train_data[:, static_idx])
        val_data_static_scaled = static_scaler.transform(val_data[:, static_idx])
        test_data_static_scaled = static_scaler.transform(test_data[:, static_idx])
        # Replace NaNs
        np.place(train_data_static_scaled, mask=np.isnan(train_data_static_scaled), vals=0.0)
        np.place(val_data_static_scaled, mask=np.isnan(val_data_static_scaled), vals=0.0)
        np.place(test_data_static_scaled, mask=np.isnan(test_data_static_scaled), vals=0.0)

        # Insert in the scaled dataset
        train_data_scaled[:, static_idx] = train_data_static_scaled
        test_data_scaled[:, static_idx] = test_data_static_scaled
        val_data_scaled[:, static_idx] = val_data_static_scaled

    # We deal with the categorical features.
    if df_ref is None:
        cat_names, cat_idx = finding_cat_features(rep_data, threshold)
    else:
        cat_names, cat_idx = finding_cat_features_fom_file(rep_data, df_ref)

    # We check for columns that are both categorical and static
    if static_idx:
        common_idx = [idx for idx in cat_idx if idx in static_idx]
        if common_idx:
            common_name = columns[common_idx]
        else:
            common_name = None

    if len(cat_names) > 0:
        # We one-hot categorical features with more than two possible values
        all_categorical_data, oh_cat_name = get_one_hot(rep_data, cat_names, cat_idx)
        if common_name is not None:
            common_cat_name = [c for c in oh_cat_name if c.split('_')[0] in common_name]
            print(common_cat_name)

        # We replace them at the end of the features
        train_data_scaled = np.concatenate([np.delete(train_data_scaled, cat_idx, axis=1),
                                            all_categorical_data['train']], axis=-1)
        test_data_scaled = np.concatenate([np.delete(test_data_scaled, cat_idx, axis=1),
                                           all_categorical_data['test']], axis=-1)
        val_data_scaled = np.concatenate([np.delete(val_data_scaled, cat_idx, axis=1),
                                          all_categorical_data['val']], axis=-1)
        columns = np.concatenate([np.delete(columns, cat_idx, axis=0), oh_cat_name], axis=0)

        # We ensure that static categorical features are also among the first features with other static ones.
        if common_name is not None:
            common_current_idx = [i for i, n in enumerate(columns) if n in common_cat_name]
            print(common_current_idx)
            new_idx = common_current_idx + [k for k in range(len(columns)) if k not in common_current_idx]
            columns = columns[new_idx]
            train_data_scaled = train_data_scaled[:, new_idx]
            test_data_scaled = test_data_scaled[:, new_idx]
            val_data_scaled = val_data_scaled[:, new_idx]

    data_dic = {'train': train_data_scaled,
                'test': test_data_scaled,
                'val': val_data_scaled}
    if 'labels' in rep_data.keys():
        label_dic = rep_data['labels']
    else:
        label_dic = None

    if 'patient_windows' in rep_data.keys():
        patient_dic = rep_data['patient_windows']
    else:
        patient_dic = None

    return data_dic, label_dic, patient_dic, columns, labels_name


def save_to_h5_with_tasks(save_path, col_names, task_names, data_dict, label_dict, patient_windows_dict):
    """
    Save a dataset with the desired format as h5.
    Args:
        save_path: Path to save the dataset to.
        col_names: List of names the variables in the dataset.
        data_dict: Dict with an array for each split of the data
        label_dict: (Optional) Dict with each split and and labels array in same order as lookup_table.
        patient_windows_dict: Dict containing a array for each split such that each row of the array is of the type
        [start_index, stop_index, patient_id].


    Returns:

    """
    with h5py.File(save_path, "w") as f:
        n_data = f.create_group('data')
        n_data.create_dataset('train', data=data_dict['train'].astype(float), dtype=np.float32)
        n_data.create_dataset('test', data=data_dict['test'].astype(float), dtype=np.float32)
        n_data.create_dataset('val', data=data_dict['val'].astype(float), dtype=np.float32)
        n_data.attrs['columns'] = list(col_names)

        if label_dict is not None:
            labels = f.create_group('labels')
            labels.create_dataset('train', data=label_dict['train'], dtype=np.float32)
            labels.create_dataset('test', data=label_dict['test'], dtype=np.float32)
            labels.create_dataset('val', data=label_dict['val'], dtype=np.float32)
            labels.attrs['tasks'] = list(task_names)

        if patient_windows_dict is not None:
            p_windows = f.create_group('patient_windows')
            p_windows.create_dataset('train', data=patient_windows_dict['train'], dtype=np.int32)
            p_windows.create_dataset('test', data=patient_windows_dict['test'], dtype=np.int32)
            p_windows.create_dataset('val', data=patient_windows_dict['val'], dtype=np.int32)

        if not len(col_names) == data_dict['train'].shape[-1]:
            raise Exception(
                "We saved to data but the number of columns ({}) didn't match the number of features {} ".format(
                    len(col_names), data_dict['train'].shape[-1]))


def build_few_label_dataset(path_to_data, path_to_save, percentage=10, seed=1234, task=None, overwrite=False):
    """Builds a dataset with reduced amounts of labeled training data.

    Stratification is made at a patient level to ensure quicker decrease in labeled data diversity.
    Args:
        path_to_data: String with path to the initial h5 file containing full dataset.
        path_to_save: String with path where to save the future dataset.
        percentage: Integer with percentage of the labeled data.
        seed: Integer with seed to split on.
        task: String with task name to stratify on.
        overwrite:

    Returns:
        Path to saved dataset.

    """
    h5py_file = h5py.File(path_to_data, 'r')
    # Processing on the label part for the evaluating part

    if task is None:
        assert ((len(h5py_file['labels']['train'].shape) == 1))
        label_idx = 0
    else:
        labels_name = h5py_file['labels'].attrs['tasks']
        label_idx = np.where(labels_name == task)[0]

    patient_windows = h5py_file['patient_windows']['train']
    train_labels = h5py_file['labels']['train']
    labels_for_stratifiction = h5py_file['labels']['train'][:, label_idx]
    train_data = h5py_file['data']['train']
    pos_patient = []
    neg_patient = []
    n_pos = {}
    n_neg = {}
    for start, stop, id_ in patient_windows:
        pos_index_patient = np.argwhere(labels_for_stratifiction[start:stop + 1] == 1).T[0]
        neg_index_patient = np.argwhere(labels_for_stratifiction[start:stop + 1] == 0).T[0]

        if len(pos_index_patient) > 0:
            pos_patient.append(id_)
            n_pos[id_] = len(pos_index_patient)
            n_neg[id_] = len(neg_index_patient)

        else:
            neg_patient.append(id_)
            n_pos[id_] = len(pos_index_patient)
            n_neg[id_] = len(neg_index_patient)

    n_pos_patient_to_take = int(len(pos_patient) // (100 / percentage))
    n_neg_patient_to_take = int(len(neg_patient) // (100 / percentage))
    print(n_pos_patient_to_take)

    rs = np.random.RandomState(seed)
    rs.shuffle(pos_patient)
    rs.shuffle(neg_patient)
    new_pos_patient = pos_patient[:n_pos_patient_to_take]
    new_neg_patient = neg_patient[:n_neg_patient_to_take]
    new_pos_patient_idx = [np.where(patient_windows[:, -1] == patient)[0][0] for patient in new_pos_patient]
    new_neg_patient_idx = [np.where(patient_windows[:, -1] == patient)[0][0] for patient in new_neg_patient]
    new_idx = np.sort(new_pos_patient_idx + new_neg_patient_idx)
    new_data = []
    new_label = []
    new_p_windows = []

    for idx in new_idx:
        start, stop, pid = patient_windows[idx]
        data = train_data[start:stop + 1]
        label = train_labels[start:stop + 1]
        if new_p_windows == []:
            new_start = 0
            new_stop = len(label) - 1
        else:
            new_start = new_p_windows[-1][1] + 1
            new_stop = new_start + len(label) - 1
        new_data.append(data)
        new_label.append(label)
        new_p_windows.append([new_start, new_stop, pid])

    lookup_table = {}
    patient_start_stop_ids = {}
    labels = {}
    lookup_table['train'] = np.concatenate(new_data, axis=0)
    lookup_table['test'] = h5py_file['data']['test'][:]
    lookup_table['val'] = h5py_file['data']['val'][:]
    patient_start_stop_ids['train'] = np.array(new_p_windows)
    patient_start_stop_ids['test'] = h5py_file['patient_windows']['test'][:]
    patient_start_stop_ids['val'] = h5py_file['patient_windows']['val'][:]
    labels['train'] = np.concatenate(new_label, axis=0)
    labels['test'] = h5py_file['labels']['test'][:]
    labels['val'] = h5py_file['labels']['val'][:]
    if task is None:
        path_to_save += path_to_data.split('/')[-1].split('.')[0] + '_{}'.format(percentage) + '_{}'.format(
            seed) + '.h5'
    else:
        path_to_save += path_to_data.split('/')[-1].split('.')[0] + '_{}'.format(percentage) + '_{}'.format(
            seed) + '_by_{}'.format(task) + '.h5'

    if os.path.isfile(path_to_save) and not overwrite:
        print('File already exist')
    else:
        with h5py.File(path_to_save, "w") as f:
            n_data = f.create_group('data')
            n_labels = f.create_group('labels')
            p_windows = f.create_group('patient_windows')
            n_data.attrs['columns'] = list(h5py_file['data'].attrs['columns'])
            n_data.attrs['seed'] = seed

            n_data.create_dataset('train', data=lookup_table['train'], dtype=np.float32)
            n_data.create_dataset('test', data=lookup_table['test'], dtype=np.float32)
            n_data.create_dataset('val', data=lookup_table['val'], dtype=np.float32)

            n_labels.create_dataset('val', data=labels['val'], dtype=np.float32)
            n_labels.create_dataset('test', data=labels['test'], dtype=np.float32)
            n_labels.create_dataset('train', data=labels['train'], dtype=np.float32)
            n_labels.attrs['tasks'] = h5py_file['labels'].attrs['tasks']

            p_windows.create_dataset('train', data=patient_start_stop_ids['train'], dtype=np.int32)
            p_windows.create_dataset('test', data=patient_start_stop_ids['test'], dtype=np.int32)
            p_windows.create_dataset('val', data=patient_start_stop_ids['val'], dtype=np.int32)

    return path_to_save
