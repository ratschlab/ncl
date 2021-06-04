import os
import random

import numpy as np
from tqdm import tqdm
from utils import clip_dataset, put_static_first


def impute_sample_1h(measures, stay_time=None):
    """Forward imputes with 1h resolution any stay to duration stay_time.

    Args:
        measures: Array-like matrix with succesives measurement.
        stay_time: (Optional) Time until which we want to impute.

    Returns:
        Imputed time-series.
    """
    forward_filled_sample = impute_sample(measures)
    imputed_sample = [np.array(forward_filled_sample[0])]
    imputed_sample[0][0] = 0
    if not stay_time:
        max_time = int(np.ceil(measures[-1, 0].astype(float)))
    else:
        max_time = int(np.ceil(stay_time))
    for k in range(1, max_time + 1):
        diff_to_k = forward_filled_sample[:, 0].astype(float) - k
        if np.argwhere(diff_to_k <= 0).shape[0] > 0:
            idx_for_k = np.argwhere(diff_to_k <= 0)[-1][0]
            time_k = np.array(forward_filled_sample[idx_for_k])
            time_k[0] = k
            imputed_sample.append(time_k)
        else:
            time_k = np.array(imputed_sample[-1])
            time_k[0] = k
            imputed_sample.append(time_k)
    imputed_sample = np.stack(imputed_sample, axis=0)

    return imputed_sample


def impute_sample(measures_t):
    """ Used under impute_sample_1h to forward impute without re-defining the resolution.

    """
    measures = np.array(measures_t)
    imputed_sample = [measures[0]]
    for k in range(1, len(measures)):
        r_t = measures[k]
        r_t_m_1 = np.array(imputed_sample[-1])
        idx_to_impute = np.argwhere(r_t == '')
        r_t[idx_to_impute] = r_t_m_1[idx_to_impute]
        imputed_sample.append(np.array(r_t))
    imputed_sample = np.stack(imputed_sample, axis=0)
    return imputed_sample


def remove_strings_col(data, col, channel_to_id, matching_dict):
    """Replaces the string arguments existing in the MIMIC-III data to category index.

    """
    transfo_data = {}
    for split in ['train', 'test', 'val']:
        current_data = np.copy(data[split])
        for channel in col:
            if channel in list(matching_dict.keys()):
                m_dict = matching_dict[channel]
                m_dict[''] = np.nan
                idx_channel = channel_to_id[channel]
                data_channel = current_data[:, idx_channel]
                r = list(map(lambda x: m_dict[x], data_channel))
                current_data[:, idx_channel] = np.array(r)
            else:
                idx_channel = channel_to_id[channel]
                data_channel = current_data[:, idx_channel]
                data_channel[np.where(data_channel == '')] = np.nan
                current_data[:, idx_channel] = data_channel.astype(float)
        transfo_data[split] = current_data.astype(float)
    return transfo_data


class Reader(object):
    """Reader class derived from https://github.com/YerevaNN/mimic3-benchmarks to read their data.

    """

    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class MultitaskReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader class derived from https://github.com/YerevaNN/mimic3-benchmarks to read their data.

        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]

        def process_ihm(x):
            return list(map(int, x.split(';')))

        def process_los(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x) // 2])), list(map(float, x[len(x) // 2:])))

        def process_ph(x):
            return list(map(int, x.split(';')))

        def process_decomp(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x) // 2])), list(map(int, x[len(x) // 2:])))

        self._data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(pheno), process_decomp(decomp))
                      for fname, t, ihm, los, pheno, decomp in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.
        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": self._data[index][1],
                "ihm": self._data[index][2],
                "los": self._data[index][3],
                "pheno": self._data[index][4],
                "decomp": self._data[index][5],
                "header": header,
                "name": name}


def extract_raw_data(base_path):
    """Wrapper around MultitaskReader to extract MIMIC-III benchmark data to our h5 format.

    Args:
        base_path: Path to source data 'data/multitask'.
        You obtain it with this command 'python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/'.

    Returns:
        data_d: Dict with data array concatenating all stays for each split.
        labels_d: Dict with labels array associate data_d for each split.
        patient_windows_d : Containing the (start, stop, patient_id) for each stay in each split.
        col: Name of the columns in data
        tasks: Name of the columns in labels
    """
    data_d = {}
    labels_d = {}
    patient_window_d = {}
    for split in ['train', 'test', 'val']:
        if split in ['train', 'val']:
            folder = os.path.join(base_path, 'train')
        else:
            folder = os.path.join(base_path, 'test')

        file = os.path.join(base_path, split + '_listfile.csv')
        sample_reader = MultitaskReader(folder, file)
        num_samples = sample_reader.get_number_of_examples()
        lookup_table = []
        start_stop_id = []
        labels_split = []
        current_idx = 0
        for idx in tqdm(range(num_samples)):
            patient_sample = sample_reader.read_example(idx)
            col = list(patient_sample['header'])
            d = patient_sample['X']
            imputed_d = impute_sample_1h(d, float(patient_sample['t']))
            patient_id = int(patient_sample['name'].split('_')[0])
            episode_nb = int(patient_sample['name'].split('_')[1][-1])
            stay_id = episode_nb * 1000000 + patient_id
            label_los = patient_sample['los']
            label_decomp = patient_sample['decomp']
            n_step = int(np.ceil(patient_sample['t']))

            # Handling of samples where LOS and Decomp masks are not same shape
            if len(patient_sample['los'][0]) > n_step:
                label_los = (patient_sample['los'][0][:n_step], patient_sample['los'][1][:n_step])
            elif len(patient_sample['los'][0]) < n_step:
                raise Exception()
            if len(patient_sample['decomp'][0]) > n_step:
                label_decomp = (patient_sample['decomp'][0][:n_step], patient_sample['decomp'][1][:n_step])

            if len(label_decomp[0]) - len(label_los[0]) != 0:
                adding = [0 for k in range(abs(len(label_decomp[0]) - len(label_los[0])))]
                new_mask = label_decomp[0] + adding
                new_labels = label_decomp[1] + adding
                label_decomp = (new_mask, new_labels)
                assert len(label_decomp[0]) - len(label_los[0]) == 0

            # We build labels in our format witih np.nan when we don't have a label
            mask_decomp, label_decomp = label_decomp
            mask_los, label_los = label_los
            mask_decomp = np.array(mask_decomp).astype(float)
            mask_los = np.array(mask_los).astype(float)
            mask_decomp[np.argwhere(mask_decomp == 0)] = np.nan
            mask_los[np.argwhere(mask_los == 0)] = np.nan
            masked_labels_los = np.concatenate([[np.nan], mask_los * np.array(label_los)], axis=0)
            masked_labels_decomp = np.concatenate([[np.nan], mask_decomp * np.array(label_decomp).astype(float)],
                                                  axis=0)
            assert imputed_d.shape[0] == masked_labels_los.shape[-1]
            lookup_table.append(imputed_d)
            start_stop_id.append([current_idx, current_idx + len(imputed_d) - 1, stay_id])
            current_idx = current_idx + len(imputed_d)
            labels_split.append(np.stack([masked_labels_los, masked_labels_decomp]))

        data_d[split] = np.concatenate(lookup_table, axis=0)
        labels_d[split] = np.concatenate(labels_split, axis=1).T
        patient_window_d[split] = np.array(start_stop_id)
        col = list(patient_sample['header'])
        tasks = ['los', 'decomp']
    return data_d, labels_d, patient_window_d, col, tasks


def benchmark_to_h5(base_path, channel_to_id, matching_dict, var_range, static_col=['Height']):
    """Wrapper around the full pre-process

    """
    data_d, labels_d, patient_window_d, col, tasks = extract_raw_data(base_path)

    no_string_data = remove_strings_col(data_d, col, channel_to_id, matching_dict)

    clipped_data = clip_dataset(var_range, no_string_data, col)

    data_inverted, col_inverted = put_static_first(clipped_data, col, static_col)

    return data_inverted, labels_d, patient_window_d, col_inverted, tasks
