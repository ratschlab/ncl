import gin
import h5py
import numpy as np
import tensorflow as tf


@gin.configurable('ICU_loader_semi_temporal')
class ICU_semi_temporal(object):
    """Data loader for ICU dataset which performs windowing on the fly for the CL and NCL task.

    Here a sample can be either only a window, or also a label for either the downstream task or the pretext task.
    """

    def __init__(self, data_path, offsets=None,
                 window_size=10, padding_size=0, on_RAM=True, step_size=1,
                 pad=True, nb_variables=-1, shuffle=True, sampling_method='ps',
                 balance=True, only_labeled=False, batch_size=1, n_sample_per_patient=1, semi=True, temporal_window=-1,
                 pad_last_batch=False, task=None):
        """

        Args:
            data_path: String representing path to the h5 data file.
            offsets: Integrer indicating by how many step we want to off-set each window.
            window_size: Integer for the temporal size of the return window.
            padding_size: Integer for optional padding before each sequence.
            on_RAM: Boolean deciding if we load the data on RAM or not.
            step_size: Integer to set the step size if we want a lower resolution.
            pad: Boolean deciding if we pad the shorter samples to window_size.
            nb_variables: Integer for the number of variable to take. -1 is all.
            shuffle: Boolean to decide whether or not we shuffle after each epoch when iterating.
            sampling_method: String either 'ps' (per sample) or 'pp' (per patient) to decide if we iterate over samples
            or patients.
            balance: List with the balancing weight for each class at classification. If boolean, False means uniform and
            True means inversly proportional to class imbalance.
            only_labeled: Boolean to decide to use only the labeled data at unsupervised training.
            batch_size: Integer for the size of each generated batch
            n_sample_per_patient: Integer representing the number of samples per patient if sampling is set to 'pp'.
            semi: Boolean to decide whether or not to generate a pretext task label corresponding to (patient_id, time)
            temporal_window: Size of window to sample from  if sampling set to 'pp'. -1 means no window.
            pad_last_batch: Boolean Whether or not to pad last batch if below batch_size.
            task: String for the name of the task associated in the h5 file.
        """

        # We set sampling config
        self.window_size = window_size
        self.pad = pad
        self.padding_size = padding_size
        self.sequence_size = window_size + padding_size
        self.offsets = offsets
        self.shuffle = shuffle
        self.sampling_method = sampling_method
        self.batch_size = batch_size
        self.n_sample_per_patient = n_sample_per_patient
        self.semi = semi
        self.temporal_window = (temporal_window * step_size) // 2
        self.pad_last_batch = pad_last_batch

        if (2 * self.temporal_window < self.n_sample_per_patient) and self.temporal_window > 0:
            raise Exception('Temporal window is to small for the number of sample per patient')

        if self.offsets is None:
            self.max_offset = 0
            self.num_offset = 0
        else:
            self.max_offset = np.max(self.offsets) * step_size
            self.num_offset = len(self.offsets)

        self.step_size = step_size
        self.data_h5 = h5py.File(data_path, "r")
        self.lookup_table = self.data_h5['data']
        if 'tasks' in list(self.data_h5['labels'].attrs.keys()):
            self.labels_name = self.data_h5['labels'].attrs['tasks']
        else:
            self.labels_name = None

        if on_RAM:
            if nb_variables < 0:
                print('Loading look up table on RAM')
                self.lookup_table = {'train': self.data_h5['data']['train'][:], 'test': self.data_h5['data']['test'][:],
                                     'val': self.data_h5['data']['val'][:]}
            else:
                print('Loading look up table on RAM with the first {} variables based on varations in time'.format(
                    nb_variables))
                self.lookup_table = {'train': self.data_h5['data']['train'][:, :nb_variables],
                                     'test': self.data_h5['data']['test'][:, :nb_variables],
                                     'val': self.data_h5['data']['val'][:, :nb_variables]}

        self.patient_start_stop_ids = {'train': self.data_h5['patient_windows']['train'][:],
                                       'test': self.data_h5['patient_windows']['test'][:],
                                       'val': self.data_h5['patient_windows']['val'][:]}

        # Processing on the label part for the evaluating part
        if task is None:
            assert ((len(self.data_h5['labels']['train'].shape) == 1) or (self.data_h5['labels'].shape[-1] == 1))
            self.label_idx = 0
        else:
            self.label_idx = np.where(self.labels_name == task)[0]

        self.labels = self.data_h5['labels']
        if on_RAM:
            if len(self.data_h5['labels']['train'].shape) > 1:
                self.labels = {'train': np.reshape(self.data_h5['labels']['train'][:, self.label_idx], (-1,)),
                               'test': np.reshape(self.data_h5['labels']['test'][:, self.label_idx], (-1,)),
                               'val': np.reshape(self.data_h5['labels']['val'][:, self.label_idx], (-1,))}
            else:
                self.labels = {'train': self.data_h5['labels']['train'][:],
                               'test': self.data_h5['labels']['test'][:],
                               'val': self.data_h5['labels']['val'][:]}

        self.valid_indexes_labels = {'train': list(np.argwhere(~np.isnan(self.labels['train'][:])).T[0]),
                                     'test': list(np.argwhere(~np.isnan(self.labels['test'][:])).T[0]),
                                     'val': list(np.argwhere(~np.isnan(self.labels['val'][:])).T[0])}

        self.num_labels = {'train': len(self.valid_indexes_labels['train']),
                           'test': len(self.valid_indexes_labels['test']),
                           'val': len(self.valid_indexes_labels['val'])}

        self.valid_indexes_us = {'train': [], 'test': [], 'val': []}
        self.valid_indexes_us_per_patient = {'train': {}, 'test': {}, 'val': {}}
        self.valid_indexes_labels_per_patient = {'train': {}, 'test': {}, 'val': {}}
        self.relative_indexes = {'train': [], 'test': [], 'val': []}
        self.idx_to_id = {'train': [], 'test': [], 'val': []}

        for split in ['train', 'test', 'val']:
            for start, stop, id_ in self.patient_start_stop_ids[split]:
                unique_id = id_
                while unique_id in self.valid_indexes_us_per_patient[split].keys():
                    unique_id += 1000000
                valid_idx = list(range(start, stop + 1 - self.max_offset))
                self.valid_indexes_us[split] += valid_idx
                if len(valid_idx) > self.n_sample_per_patient:
                    self.valid_indexes_us_per_patient[split][unique_id] = np.array(valid_idx)
                correct_labels = list(np.argwhere(~np.isnan(self.labels[split][start:stop + 1])).T[0] + start)
                if correct_labels:
                    self.valid_indexes_labels_per_patient[split][unique_id] = np.array(correct_labels)
                self.relative_indexes[split] += list(range(0, stop + 1 - start))
                self.idx_to_id[split] += list(np.ones((stop + 1 - start,)) * unique_id)

        if only_labeled:
            self.valid_indexes_us = self.valid_indexes_labels
            self.valid_indexes_us_per_patient = self.valid_indexes_labels_per_patient

        for split in ['train', 'test', 'val']:
            self.relative_indexes[split] = np.array(self.relative_indexes[split])
            self.valid_indexes_us[split] = np.array(self.valid_indexes_us[split])
            self.valid_indexes_labels[split] = np.array(self.valid_indexes_labels[split])
            self.idx_to_id[split] = np.array(self.idx_to_id[split])

        self.num_samples = {'train': len(self.valid_indexes_us['train']), 'test': len(self.valid_indexes_us['test']),
                            'val': len(self.valid_indexes_us['val'])}

        # Weights balance fort labels:
        if balance:
            if isinstance(balance, bool):
                pos_num_val = np.nansum(self.labels['val'][:]) / self.num_labels['val']
                pos_num_test = np.nansum(self.labels['test'][:]) / self.num_labels['test']
                pos_num_train = np.nansum(self.labels['train'][:]) / self.num_labels['train']
                self.labels_weights = {'train': [0.5 / (1 - pos_num_train), 0.5 / pos_num_train],
                                       'test': [0.5 / (1 - pos_num_test), 0.5 / pos_num_test],
                                       'val': [0.5 / (1 - pos_num_val), 0.5 / pos_num_val]}
            else:
                self.labels_weights = {'train': balance,
                                       'test': balance,
                                       'val': balance}


        else:
            self.labels_weights = {'train': [1.0, 1.0],
                                   'test': [1.0, 1.0],
                                   'val': [1.0, 1.0]}
        # Iterate counters
        self.current_index_training = {'train': 0, 'test': 0, 'val': 0}
        self.current_index_evaluating = {'train': 0, 'test': 0, 'val': 0}
        self.bias_init = [0.0, np.nansum(self.labels['train'][:]) / self.num_labels['train'] / (
                1 - np.nansum(self.labels['train'][:]) / self.num_labels['train'])]

        # Per patient variables
        self.valid_id_eval = {'train': np.array(list(self.valid_indexes_labels_per_patient['train'].keys())),
                              'test': np.array(list(self.valid_indexes_labels_per_patient['test'].keys())),
                              'val': np.array(list(self.valid_indexes_labels_per_patient['val'].keys()))}

        self.valid_id_us = {'train': np.array(list(self.valid_indexes_us_per_patient['train'].keys())),
                            'test': np.array(list(self.valid_indexes_us_per_patient['test'].keys())),
                            'val': np.array(list(self.valid_indexes_us_per_patient['val'].keys()))}

        self.num_patient_eval = {'train': len(self.valid_id_eval['train']),
                                 'test': len(self.valid_id_eval['test']),
                                 'val': len(self.valid_id_eval['val'])}

        self.num_patient_us = {'train': len(self.valid_id_us['train']),
                               'test': len(self.valid_id_us['test']),
                               'val': len(self.valid_id_us['val'])}

    def sort_indexes(self):
        """
        Function to resort the indexes at evaluation if needed.
        Returns:

        """
        for split in ['train', 'test', 'val']:
            self.valid_indexes_labels[split] = np.sort(self.valid_indexes_labels[split])
            self.valid_indexes_us[split] = np.sort(self.valid_indexes_us[split])

    def sample(self, random_state, split='train', training=False, idx_patient=None):
        """Function to sample from the data split of choice.

        This methods is further wrapped into a generator to build a tf.data.Dataset

        Args:
            random_state: np.random.RandomState instance to for the idx choice.
            split: String representing split to sample from, either 'train' or 'test'.
            training: Boolean to determine if we are training the representation.
            idx_patient: (Optional) array  to sample particular samples from given  indexes.

        Returns:
            A batch from the desired distribution as tuple of numpy arrays if semi or not training. Otherwise a array.
        """

        # If we are training the representation
        if training:
            if idx_patient is None:
                if self.sampling_method == 'ps':
                    idx_patient = random_state.randint(self.num_samples[split], size=(self.batch_size,))
                    state_idx = self.valid_indexes_us[split][idx_patient]

                elif self.sampling_method == 'pp':
                    idx_id = random_state.choice(np.arange(self.num_patient_us[split]), replace=False,
                                                 size=(int(np.ceil(self.batch_size / self.n_sample_per_patient)),))
                    idx_id = self.valid_id_us[split][idx_id]
                    possible_states = [self.valid_indexes_us_per_patient[split].get(id_) for id_ in idx_id]

                    if self.temporal_window > 0:

                        anchor_limits_states = []
                        for possible_state in possible_states:
                            if len(possible_state) > 2 * self.temporal_window + 1:
                                center = random_state.randint(self.temporal_window, len(
                                    possible_state) - self.temporal_window, size=(1,))
                            else:
                                center = len(possible_state) // 2

                            anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                        anchor_limits_states = np.array(anchor_limits_states)

                        lower_neighbor = np.max(
                            [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                        higher_neighbor = np.min(
                            [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)

                        state_idx = np.concatenate(
                            [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,))
                             for
                             low, high in
                             zip(lower_neighbor, higher_neighbor)])
                    else:
                        state_idx = np.concatenate(
                            [random_state.choice(pos_state, replace=False, size=(self.n_sample_per_patient,))
                             for pos_state in possible_states])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

                else:
                    raise Exception("Sampling methods is either 'pp' or 'ps'")
            else:
                if self.sampling_method == 'ps':
                    state_idx = np.reshape(self.valid_indexes_us[split][idx_patient], (-1,))

                elif self.sampling_method == 'pp':
                    idx_id = self.valid_id_us[split][idx_patient]
                    possible_states = [self.valid_indexes_us_per_patient[split].get(id_) for id_ in idx_id]
                    if self.temporal_window > 0:

                        anchor_limits_states = []
                        for possible_state in possible_states:
                            if len(possible_state) > 2 * self.temporal_window + 1:
                                center = random_state.randint(self.temporal_window, len(
                                    possible_state) - self.temporal_window, size=(1,))
                            else:
                                center = len(possible_state) // 2

                            anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                        anchor_limits_states = np.array(anchor_limits_states)

                        lower_neighbor = np.max(
                            [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                        higher_neighbor = np.min(
                            [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)

                        if np.any(lower_neighbor >= higher_neighbor):
                            print(lower_neighbor, higher_neighbor)
                        state_idx = np.concatenate(
                            [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,))
                             for
                             low, high in
                             zip(lower_neighbor, higher_neighbor)])
                    else:
                        state_idx = np.concatenate(
                            [random_state.choice(pos_state, replace=False, size=(self.n_sample_per_patient,)) for
                             pos_state in possible_states])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

            y = np.stack([self.idx_to_id[split][state_idx], state_idx]).T // self.step_size
        else:
            if idx_patient is None:
                if self.sampling_method == 'ps':
                    idx_patient = random_state.randint(self.num_labels[split], size=(self.batch_size,))
                    state_idx = self.valid_indexes_labels[split][idx_patient]

                elif self.sampling_method == 'pp':
                    idx_id = random_state.choice(np.arange(self.num_patient_eval[split]), replace=False,
                                                 size=(int(np.ceil(self.batch_size / self.n_sample_per_patient)),))
                    idx_id = self.valid_id_eval[split][idx_id]
                    possible_states = [self.valid_indexes_labels_per_patient[split].get(id_) for id_ in idx_id]

                    anchor_limits_states = []
                    for possible_state in possible_states:
                        if len(possible_state) > 2 * self.temporal_window + 1:
                            center = random_state.randint(self.temporal_window, len(
                                possible_state) - self.temporal_window, size=(1,))
                        else:
                            center = len(possible_state) // 2

                        anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                    anchor_limits_states = np.array(anchor_limits_states)

                    lower_neighbor = np.max(
                        [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                    higher_neighbor = np.min(
                        [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)
                    state_idx = np.concatenate(
                        [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,)) for
                         low, high in
                         zip(lower_neighbor, higher_neighbor)])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

                else:
                    raise Exception("Sampling methods is either 'pp' or 'ps'")
            else:
                if self.sampling_method == 'ps':
                    state_idx = np.reshape(self.valid_indexes_labels[split][idx_patient], (-1,))

                elif self.sampling_method == 'pp':
                    idx_id = self.valid_id_eval[split][idx_patient]

                    possible_states = [self.valid_indexes_labels_per_patient[split].get(id_) for id_ in idx_id]

                    anchor_limits_states = []
                    for possible_state in possible_states:
                        if len(possible_state) > 2 * self.temporal_window + 1:
                            center = random_state.randint(self.temporal_window, len(
                                possible_state) - self.temporal_window, size=(1,))
                        else:
                            center = len(possible_state) // 2

                        anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                    anchor_limits_states = np.array(anchor_limits_states)
                    lower_neighbor = np.max(
                        [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                    higher_neighbor = np.min(
                        [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)
                    state_idx = np.concatenate(
                        [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,)) for
                         low, high in
                         zip(lower_neighbor, higher_neighbor)])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

            y = self.labels[split][state_idx]

        relative_idx = self.relative_indexes[split][state_idx]
        start = state_idx - relative_idx
        relative_start = relative_idx % self.step_size + start
        relative_idx = relative_idx // self.step_size
        windowing_inputs = np.stack([state_idx, relative_start, relative_idx], axis=-1)
        X = np.array(list(
            map(lambda x: self.windowing(x[0], x[1], x[2], self.offsets, self.window_size, self.padding_size, split,
                                         self.step_size,
                                         self.pad), windowing_inputs)))
        if self.semi or (not training):
            return X, y
        else:
            return X

    def iterate(self, random_state, split='train', training=False):
        """Function to iterate over the data split of choice.

        This methods is further wrapped into a generator to build a tf.data.Dataset.
        It wraps around the sample method.

        Args:
            random_state: np.random.RandomState instance to for the idx choice.
            split: String representing split to sample from, either 'train', 'test' or 'val'.
            training: Boolean to determine if we are training the representation.

        Returns:
            A sample corresponding to the current_index from the desired distribution as tuple of numpy arrays.
        """
        if training:
            if self.sampling_method == 'ps':
                if (self.current_index_training[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_indexes_us[split])

                next_idx = list(
                    range(self.current_index_training[split], self.current_index_training[split] + self.batch_size))
                self.current_index_training[split] += self.batch_size

                if self.current_index_training[split] >= self.num_samples[split]:
                    n_exceeding_samples = self.current_index_training[split] - self.num_samples[split]
                    assert n_exceeding_samples <= self.batch_size
                    if not self.pad_last_batch:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_training[split] = 0
            else:

                if (self.current_index_training[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_id_us[split])
                n_patient = int(np.ceil(self.batch_size / self.n_sample_per_patient))
                next_idx = list(
                    range(self.current_index_training[split], self.current_index_training[split] + n_patient))
                self.current_index_training[split] += n_patient

                if self.current_index_training[split] >= self.num_patient_us[split]:
                    n_exceeding_samples = self.current_index_training[split] - self.num_patient_us[split]
                    assert n_exceeding_samples <= n_patient
                    if not self.pad_last_batch:
                        next_idx = next_idx[:n_patient - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:n_patient - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_training[split] = 0

            sample = self.sample(random_state, split, training, idx_patient=next_idx)

            return sample

        else:
            if self.sampling_method == 'ps':
                if (self.current_index_evaluating[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_indexes_labels[split])

                next_idx = list(
                    range(self.current_index_evaluating[split], self.current_index_evaluating[split] + self.batch_size))
                self.current_index_evaluating[split] += self.batch_size

                if self.current_index_evaluating[split] >= self.num_labels[split]:
                    n_exceeding_samples = self.current_index_evaluating[split] - self.num_labels[split]
                    assert n_exceeding_samples <= self.batch_size
                    if not self.pad_last_batch:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_evaluating[split] = 0
            else:
                if (self.current_index_evaluating[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_id_eval[split])
                n_patient = int(np.ceil(self.batch_size / self.n_sample_per_patient))
                next_idx = list(
                    range(self.current_index_evaluating[split], self.current_index_evaluating[split] + n_patient))
                self.current_index_evaluating[split] += n_patient

                if self.current_index_evaluating[split] >= self.num_patient_eval[split]:
                    n_exceeding_samples = self.current_index_evaluating[split] - self.num_patient_eval[split]
                    assert n_exceeding_samples <= n_patient
                    if not self.pad_last_batch:
                        next_idx = next_idx[:n_patient - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:n_patient - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_evaluating[split] = 0

            sample = self.sample(random_state, split, training, idx_patient=next_idx)
            return sample

    def windowing(self, state_idx, relative_start, relative_idx, offsets=None, window_size=200, padding_size=0,
                  split='train', step_size=1, pad=True):
        """Function extracting a window correponding a state index.

        Args:
            state_idx: Integer representing the state index in the lookup-table.
            relative_start: Integer representing the start of the patient stay.
            relative_idx: Integer representing the (end of the window)/(index of the state) in the stay.
            offsets: Integer setting the number of off-sets for the window if needed.
            window_size: Integer of the size of the window to sample.
            padding_size: Integer for size of the padding to add to all windows.
            split: String with the split of the data to take the window from.
            step_size: Integer with relative step size for smaller resolution windows.
            pad: Boolean for whether or to pad shorter windows to window_size.

        Returns:
                A window  as numpy array of shape (num_features, padding_size + len(offsets) + window_size).
        """

        if relative_idx + 1 < window_size:
            seq = self.lookup_table[split][relative_start: state_idx + 1: step_size]
            if pad:
                seq = np.concatenate(
                    [np.zeros((padding_size + window_size - relative_idx - 1, self.lookup_table[split].shape[-1])),
                     seq], axis=0)
        else:
            seq = self.lookup_table[split][state_idx + step_size - window_size * step_size:state_idx + 1:step_size]
            if padding_size != 0 and pad:
                seq = np.concatenate([-np.ones((padding_size, self.lookup_table[split].shape[-1])), seq])

        if offsets is not None:
            off_sets = [self.lookup_table[split][state_idx + k * step_size] for k in offsets]
            seq = np.concatenate([seq, off_sets], axis=0)

        return seq

    def sample_shape(self, training=False, split='train'):
        if not training:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1]), (
                None,)
        elif self.semi:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1]), (
                None, 2)
        else:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1])

    def sample_type(self, training=False):
        if not training or self.semi:
            return tf.float32, tf.float32
        else:
            return tf.float32


@gin.configurable('ICU_loader_SCL')
class ICU_supervised_CL(object):
    """Data loader for ICU dataset which performs windowing on the fly for the SCL task.

    Here a sample can be either only be a tuple (window, task label).
    """

    def __init__(self, data_path, offsets=None,
                 window_size=10, padding_size=0, on_RAM=True, step_size=1,
                 pad=True, nb_variables=-1, shuffle=True, sampling_method='ps',
                 balance=True, only_labeled=False, batch_size=1, n_sample_per_patient=1, semi=True, temporal_window=-1,
                 pad_last_batch=False, task=None):
        """

        Args:
            data_path: String representing path to the h5 data file.
            offsets: Integrer indicating by how many step we want to off-set each window.
            window_size: Integer for the temporal size of the return window.
            padding_size: Integer for optional padding before each sequence.
            on_RAM: Boolean deciding if we load the data on RAM or not.
            step_size: Integer to set the step size if we want a lower resolution.
            pad: Boolean deciding if we pad the shorter samples to window_size.
            nb_variables: Integer for the number of variable to take. -1 is all.
            shuffle: Boolean to decide whether or not we shuffle after each epoch when iterating.
            sampling_method: String either 'ps' (per sample) or 'pp' (per patient) to decide if we iterate over samples
            or patients.
            balance: List with the balancing weight for each class at classification. If boolean, False means uniform and
            True means inversly proportional to class imbalance.
            only_labeled: Boolean to decide to use only the labeled data at unsupervised training.
            batch_size: Integer for the size of each generated batch
            n_sample_per_patient: Integer representing the number of samples per patient if sampling is set to 'pp'.
            semi: Boolean to decide whether or not to generate a pretext task label corresponding to (patient_id, time)
            temporal_window: Size of window to sample from  if sampling set to 'pp'. -1 means no window.
            pad_last_batch: Bolean Whether or not to pad last batch if below batch_size.
            task: String for the name of the task associated in the h5 file.
        """

        # We set sampling config
        self.window_size = window_size
        self.pad = pad
        self.padding_size = padding_size
        self.sequence_size = window_size + padding_size
        self.offsets = offsets
        self.shuffle = shuffle
        self.sampling_method = sampling_method
        self.batch_size = batch_size
        self.n_sample_per_patient = n_sample_per_patient
        self.semi = semi
        self.temporal_window = (temporal_window * step_size) // 2
        self.pad_last_batch = pad_last_batch

        if (2 * self.temporal_window < self.n_sample_per_patient) and self.temporal_window > 0:
            raise Exception('Temporal window is to small for the number of sample per patient')

        if self.offsets is None:
            self.max_offset = 0
            self.num_offset = 0
        else:
            self.max_offset = np.max(self.offsets) * step_size
            self.num_offset = len(self.offsets)

        self.step_size = step_size
        self.data_h5 = h5py.File(data_path, "r")
        self.lookup_table = self.data_h5['data']
        if 'tasks' in list(self.data_h5['labels'].attrs.keys()):
            self.labels_name = self.data_h5['labels'].attrs['tasks']
        else:
            self.labels_name = None

        if on_RAM:
            if nb_variables < 0:
                print('Loading look up table on RAM')
                self.lookup_table = {'train': self.data_h5['data']['train'][:], 'test': self.data_h5['data']['test'][:],
                                     'val': self.data_h5['data']['val'][:]}
            else:
                print('Loading look up table on RAM witht the first {} variables based on varations in time'.format(
                    nb_variables))
                self.lookup_table = {'train': self.data_h5['data']['train'][:, :nb_variables],
                                     'test': self.data_h5['data']['test'][:, :nb_variables],
                                     'val': self.data_h5['data']['val'][:, :nb_variables]}

        self.patient_start_stop_ids = {'train': self.data_h5['patient_windows']['train'][:],
                                       'test': self.data_h5['patient_windows']['test'][:],
                                       'val': self.data_h5['patient_windows']['val'][:]}

        # Processing on the label part for the evaluating part
        if task is None:
            assert ((len(self.data_h5['labels']['train'].shape) == 1) or (self.data_h5['labels'].shape[-1] == 1))
            self.label_idx = 0
        else:
            self.label_idx = np.where(self.labels_name == task)[0]
        self.labels = self.data_h5['labels']
        if on_RAM:
            if len(self.data_h5['labels']['train'].shape) > 1:
                self.labels = {'train': np.reshape(self.data_h5['labels']['train'][:, self.label_idx], (-1,)),
                               'test': np.reshape(self.data_h5['labels']['test'][:, self.label_idx], (-1,)),
                               'val': np.reshape(self.data_h5['labels']['val'][:, self.label_idx], (-1,))}
            else:
                self.labels = {'train': self.data_h5['labels']['train'][:],
                               'test': self.data_h5['labels']['test'][:],
                               'val': self.data_h5['labels']['val'][:]}

        self.valid_indexes_labels = {'train': list(np.argwhere(~np.isnan(self.labels['train'][:])).T[0]),
                                     'test': list(np.argwhere(~np.isnan(self.labels['test'][:])).T[0]),
                                     'val': list(np.argwhere(~np.isnan(self.labels['val'][:])).T[0])}
        self.num_labels = {'train': len(self.valid_indexes_labels['train']),
                           'test': len(self.valid_indexes_labels['test']),
                           'val': len(self.valid_indexes_labels['val'])}

        self.valid_indexes_us = {'train': [], 'test': [], 'val': []}
        self.valid_indexes_us_per_patient = {'train': {}, 'test': {}, 'val': {}}
        self.valid_indexes_labels_per_patient = {'train': {}, 'test': {}, 'val': {}}
        self.relative_indexes = {'train': [], 'test': [], 'val': []}
        self.idx_to_id = {'train': [], 'test': [], 'val': []}
        for split in ['train', 'test', 'val']:
            for start, stop, id_ in self.patient_start_stop_ids[split]:
                unique_id = id_
                while unique_id in self.valid_indexes_us_per_patient[split].keys():
                    unique_id += 1000000
                valid_idx = list(range(start, stop + 1 - self.max_offset))
                self.valid_indexes_us[split] += valid_idx
                if len(valid_idx) > self.n_sample_per_patient:
                    self.valid_indexes_us_per_patient[split][unique_id] = np.array(valid_idx)
                correct_labels = list(np.argwhere(~np.isnan(self.labels[split][start:stop + 1])).T[0] + start)
                if correct_labels:
                    self.valid_indexes_labels_per_patient[split][unique_id] = np.array(correct_labels)
                self.relative_indexes[split] += list(range(0, stop + 1 - start))
                self.idx_to_id[split] += list(np.ones((stop + 1 - start,)) * unique_id)

        if only_labeled:
            self.valid_indexes_us = self.valid_indexes_labels
            self.valid_indexes_us_per_patient = self.valid_indexes_labels_per_patient

        for split in ['train', 'test', 'val']:
            self.relative_indexes[split] = np.array(self.relative_indexes[split])
            self.valid_indexes_us[split] = np.array(self.valid_indexes_us[split])
            self.valid_indexes_labels[split] = np.array(self.valid_indexes_labels[split])
            self.idx_to_id[split] = np.array(self.idx_to_id[split])

        self.num_samples = {'train': len(self.valid_indexes_us['train']), 'test': len(self.valid_indexes_us['test']),
                            'val': len(self.valid_indexes_us['val'])}

        # Weights balance fort labels:
        if balance:
            if isinstance(balance, bool):
                pos_num_val = np.nansum(self.labels['val'][:]) / self.num_labels['val']
                pos_num_test = np.nansum(self.labels['test'][:]) / self.num_labels['test']
                pos_num_train = np.nansum(self.labels['train'][:]) / self.num_labels['train']
                self.labels_weights = {'train': [0.5 / (1 - pos_num_train), 0.5 / pos_num_train],
                                       'test': [0.5 / (1 - pos_num_test), 0.5 / pos_num_test],
                                       'val': [0.5 / (1 - pos_num_val), 0.5 / pos_num_val]}
            else:
                self.labels_weights = {'train': balance,
                                       'test': balance,
                                       'val': balance}


        else:
            self.labels_weights = {'train': [1.0, 1.0],
                                   'test': [1.0, 1.0],
                                   'val': [1.0, 1.0]}
        # Iterate counters
        self.current_index_training = {'train': 0, 'test': 0, 'val': 0}
        self.current_index_evaluating = {'train': 0, 'test': 0, 'val': 0}

        self.bias_init = [0.0, np.nansum(self.labels['train'][:]) / self.num_labels['train'] / (
                1 - np.nansum(self.labels['train'][:]) / self.num_labels['train'])]

        # Per patient variables
        self.valid_id_eval = {'train': np.array(list(self.valid_indexes_labels_per_patient['train'].keys())),
                              'test': np.array(list(self.valid_indexes_labels_per_patient['test'].keys())),
                              'val': np.array(list(self.valid_indexes_labels_per_patient['val'].keys()))}

        self.valid_id_us = {'train': np.array(list(self.valid_indexes_us_per_patient['train'].keys())),
                            'test': np.array(list(self.valid_indexes_us_per_patient['test'].keys())),
                            'val': np.array(list(self.valid_indexes_us_per_patient['val'].keys()))}

        self.num_patient_eval = {'train': len(self.valid_id_eval['train']),
                                 'test': len(self.valid_id_eval['test']),
                                 'val': len(self.valid_id_eval['val'])}

        self.num_patient_us = {'train': len(self.valid_id_us['train']),
                               'test': len(self.valid_id_us['test']),
                               'val': len(self.valid_id_us['val'])}

    def sort_indexes(self):
        """
        Function to resort the indexes at evaluation if needed.
        Returns:

        """
        for split in ['train', 'test', 'val']:
            self.valid_indexes_labels[split] = np.sort(self.valid_indexes_labels[split])
            self.valid_indexes_us[split] = np.sort(self.valid_indexes_us[split])

    def sample(self, random_state, split='train', training=False, idx_patient=None):
        """Function to sample from the data split of choice.

        This methods is further wrapped into a generator to build a tf.data.Dataset

        Args:
            random_state: np.random.RandomState instance to for the idx choice.
            split: String representing split to sample from, either 'train' or 'test'.
            training: Boolean to determine if we are training the representation.
            idx_patient: (Optional) array  to sample particular samples from given  indexes.

        Returns:
            A batch from the desired distribution as tuple of numpy arrays if semi or not training. Otherwise a array.
        """

        if training:
            if idx_patient is None:
                if self.sampling_method == 'ps':
                    idx_patient = random_state.randint(self.num_samples[split], size=(self.batch_size,))
                    state_idx = self.valid_indexes_us[split][idx_patient]

                elif self.sampling_method == 'pp':
                    idx_id = random_state.choice(np.arange(self.num_patient_us[split]), replace=False,
                                                 size=(int(np.ceil(self.batch_size / self.n_sample_per_patient)),))
                    idx_id = self.valid_id_us[split][idx_id]
                    possible_states = [self.valid_indexes_us_per_patient[split].get(id_) for id_ in idx_id]

                    if self.temporal_window > 0:
                        anchor_limits_states = []
                        for possible_state in possible_states:
                            if len(possible_state) > 2 * self.temporal_window + 1:
                                center = random_state.randint(self.temporal_window, len(
                                    possible_state) - self.temporal_window, size=(1,))
                            else:
                                center = len(possible_state) // 2

                            anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                        anchor_limits_states = np.array(anchor_limits_states)

                        lower_neighbor = np.max(
                            [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                        higher_neighbor = np.min(
                            [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)

                        state_idx = np.concatenate(
                            [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,))
                             for
                             low, high in
                             zip(lower_neighbor, higher_neighbor)])
                    else:
                        state_idx = np.concatenate(
                            [random_state.choice(pos_state, replace=False, size=(self.n_sample_per_patient,))
                             for pos_state in possible_states])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

                else:
                    raise Exception("Sampling methods is either 'pp' or 'ps'")
            else:
                if self.sampling_method == 'ps':
                    state_idx = np.reshape(self.valid_indexes_us[split][idx_patient], (-1,))

                elif self.sampling_method == 'pp':
                    idx_id = self.valid_id_us[split][idx_patient]
                    possible_states = [self.valid_indexes_us_per_patient[split].get(id_) for id_ in idx_id]
                    if self.temporal_window > 0:

                        anchor_limits_states = []
                        for possible_state in possible_states:
                            if len(possible_state) > 2 * self.temporal_window + 1:
                                center = random_state.randint(self.temporal_window, len(
                                    possible_state) - self.temporal_window, size=(1,))
                            else:
                                center = len(possible_state) // 2

                            anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                        anchor_limits_states = np.array(anchor_limits_states)

                        lower_neighbor = np.max(
                            [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                        higher_neighbor = np.min(
                            [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)

                        if np.any(lower_neighbor >= higher_neighbor):
                            print(lower_neighbor, higher_neighbor)
                        state_idx = np.concatenate(
                            [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,))
                             for
                             low, high in
                             zip(lower_neighbor, higher_neighbor)])
                    else:
                        state_idx = np.concatenate(
                            [random_state.choice(pos_state, replace=False, size=(self.n_sample_per_patient,)) for
                             pos_state in possible_states])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

            y = self.labels[split][state_idx]
        else:
            if idx_patient is None:
                if self.sampling_method == 'ps':
                    idx_patient = random_state.randint(self.num_labels[split], size=(self.batch_size,))
                    state_idx = self.valid_indexes_labels[split][idx_patient]

                elif self.sampling_method == 'pp':
                    idx_id = random_state.choice(np.arange(self.num_patient_eval[split]), replace=False,
                                                 size=(int(np.ceil(self.batch_size / self.n_sample_per_patient)),))
                    idx_id = self.valid_id_eval[split][idx_id]
                    possible_states = [self.valid_indexes_labels_per_patient[split].get(id_) for id_ in idx_id]

                    anchor_limits_states = []
                    for possible_state in possible_states:
                        if len(possible_state) > 2 * self.temporal_window + 1:
                            center = random_state.randint(self.temporal_window, len(
                                possible_state) - self.temporal_window, size=(1,))
                        else:
                            center = len(possible_state) // 2

                        anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                    anchor_limits_states = np.array(anchor_limits_states)

                    lower_neighbor = np.max(
                        [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                    higher_neighbor = np.min(
                        [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)
                    state_idx = np.concatenate(
                        [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,)) for
                         low, high in
                         zip(lower_neighbor, higher_neighbor)])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

                else:
                    raise Exception("Sampling methods is either 'pp' or 'ps'")
            else:
                if self.sampling_method == 'ps':
                    state_idx = np.reshape(self.valid_indexes_labels[split][idx_patient], (-1,))

                elif self.sampling_method == 'pp':
                    idx_id = self.valid_id_eval[split][idx_patient]

                    possible_states = [self.valid_indexes_labels_per_patient[split].get(id_) for id_ in idx_id]

                    anchor_limits_states = []
                    for possible_state in possible_states:
                        if len(possible_state) > 2 * self.temporal_window + 1:
                            center = random_state.randint(self.temporal_window, len(
                                possible_state) - self.temporal_window, size=(1,))
                        else:
                            center = len(possible_state) // 2

                        anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                    anchor_limits_states = np.array(anchor_limits_states)
                    lower_neighbor = np.max(
                        [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                    higher_neighbor = np.min(
                        [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)
                    state_idx = np.concatenate(
                        [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,)) for
                         low, high in
                         zip(lower_neighbor, higher_neighbor)])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

            y = self.labels[split][state_idx]

        relative_idx = self.relative_indexes[split][state_idx]
        start = state_idx - relative_idx
        relative_start = relative_idx % self.step_size + start
        relative_idx = relative_idx // self.step_size
        windowing_inputs = np.stack([state_idx, relative_start, relative_idx], axis=-1)
        X = np.array(list(
            map(lambda x: self.windowing(x[0], x[1], x[2], self.offsets, self.window_size, self.padding_size, split,
                                         self.step_size,
                                         self.pad), windowing_inputs)))
        if self.semi or (not training):
            return X, y
        else:
            return X

    def iterate(self, random_state, split='train', training=False):
        """Function to iterate over the data split of choice.

        This methods is further wrapped into a generator to build a tf.data.Dataset.
        It wraps around the sample method.

        Args:
            random_state: np.random.RandomState instance to for the idx choice.
            split: String representing split to sample from, either 'train', 'test' or 'val'.
            training: Boolean to determine if we are training the representation.

        Returns:
            A sample corresponding to the current_index from the desired distribution as tuple of numpy arrays.
        """
        if training:
            if self.sampling_method == 'ps':
                if (self.current_index_training[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_indexes_us[split])

                next_idx = list(
                    range(self.current_index_training[split], self.current_index_training[split] + self.batch_size))
                self.current_index_training[split] += self.batch_size

                if self.current_index_training[split] >= self.num_samples[split]:
                    n_exceeding_samples = self.current_index_training[split] - self.num_samples[split]
                    assert n_exceeding_samples <= self.batch_size
                    if not self.pad_last_batch:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_training[split] = 0
            else:

                if (self.current_index_training[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_id_us[split])
                n_patient = int(np.ceil(self.batch_size / self.n_sample_per_patient))
                next_idx = list(
                    range(self.current_index_training[split], self.current_index_training[split] + n_patient))
                self.current_index_training[split] += n_patient

                if self.current_index_training[split] >= self.num_patient_us[split]:
                    n_exceeding_samples = self.current_index_training[split] - self.num_patient_us[split]
                    assert n_exceeding_samples <= n_patient
                    if not self.pad_last_batch:
                        next_idx = next_idx[:n_patient - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:n_patient - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_training[split] = 0

            sample = self.sample(random_state, split, training, idx_patient=next_idx)

            return sample

        else:
            if self.sampling_method == 'ps':
                if (self.current_index_evaluating[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_indexes_labels[split])

                next_idx = list(
                    range(self.current_index_evaluating[split], self.current_index_evaluating[split] + self.batch_size))
                self.current_index_evaluating[split] += self.batch_size

                if self.current_index_evaluating[split] >= self.num_labels[split]:
                    n_exceeding_samples = self.current_index_evaluating[split] - self.num_labels[split]
                    assert n_exceeding_samples <= self.batch_size
                    if not self.pad_last_batch:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_evaluating[split] = 0
            else:
                if (self.current_index_evaluating[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_id_eval[split])
                n_patient = int(np.ceil(self.batch_size / self.n_sample_per_patient))
                next_idx = list(
                    range(self.current_index_evaluating[split], self.current_index_evaluating[split] + n_patient))
                self.current_index_evaluating[split] += n_patient

                if self.current_index_evaluating[split] >= self.num_patient_eval[split]:
                    n_exceeding_samples = self.current_index_evaluating[split] - self.num_patient_eval[split]
                    assert n_exceeding_samples <= n_patient
                    if not self.pad_last_batch:
                        next_idx = next_idx[:n_patient - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:n_patient - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_evaluating[split] = 0

            sample = self.sample(random_state, split, training, idx_patient=next_idx)
            return sample

    def windowing(self, state_idx, relative_start, relative_idx, offsets=None, window_size=200, padding_size=0,
                  split='train', step_size=1, pad=True):
        """Function extracting a window correponding a state index.

        Args:
            state_idx: Integer representing the state index in the lookup-table.
            relative_start: Integer representing the start of the patient stay.
            relative_idx: Integer representing the (end of the window)/(index of the state) in the stay.
            offsets: Integer setting the number of off-sets for the window if needed.
            window_size: Integer of the size of the window to sample.
            padding_size: Integer for size of the padding to add to all windows.
            split: String with the split of the data to take the window from.
            step_size: Integer with relative step size for smaller resolution windows.
            pad: Boolean for whether or to pad shorter windows to window_size.

        Returns:
                A window  as numpy array of shape (num_features, padding_size + len(offsets) + window_size).
        """

        if relative_idx + 1 < window_size:
            seq = self.lookup_table[split][relative_start: state_idx + 1: step_size]
            if pad:
                seq = np.concatenate(
                    [np.zeros((padding_size + window_size - relative_idx - 1, self.lookup_table[split].shape[-1])),
                     seq], axis=0)
        else:
            seq = self.lookup_table[split][state_idx + step_size - window_size * step_size:state_idx + 1:step_size]
            if padding_size != 0 and pad:
                seq = np.concatenate([-np.ones((padding_size, self.lookup_table[split].shape[-1])), seq])

        if offsets is not None:
            off_sets = [self.lookup_table[split][state_idx + k * step_size] for k in offsets]
            seq = np.concatenate([seq, off_sets], axis=0)

        return seq

    def sample_shape(self, training=False, split='train'):
        if not training:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1]), (
                None,)
        elif self.semi:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1]), (
                None,)
        else:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1])

    def sample_type(self, training=False):
        if not training or self.semi:
            return tf.float32, tf.float32
        else:
            return tf.float32


@gin.configurable('ICU_loader_omega_SCL')
class ICU_omega_SCL(object):
    """Data loader for ICU dataset which performs windowing on the fly for the NCL union task.

    """

    def __init__(self, data_path, offsets=None,
                 window_size=10, padding_size=0, on_RAM=True, step_size=1,
                 pad=True, nb_variables=-1, shuffle=True, sampling_method='ps',
                 balance=True, only_labeled=False, batch_size=1, n_sample_per_patient=1, semi=True, temporal_window=-1,
                 pad_last_batch=False, task=None):
        """

        Args:
            data_path: String representing path to the h5 data file.
            offsets: Integrer indicating by how many step we want to off-set each window.
            window_size: Integer for the temporal size of the return window.
            padding_size: Integer for optional padding before each sequence.
            on_RAM: Boolean deciding if we load the data on RAM or not.
            step_size: Integer to set the step size if we want a lower resolution.
            pad: Boolean deciding if we pad the shorter samples to window_size.
            nb_variables: Integer for the number of variable to take. -1 is all.
            shuffle: Boolean to decide whether or not we shuffle after each epoch when iterating.
            sampling_method: String either 'ps' (per sample) or 'pp' (per patient) to decide if we iterate over samples
            or patients.
            balance: List with the balancing weight for each class at classification. If boolean, False means uniform and
            True means inversly proportional to class imbalance.
            only_labeled: Boolean to decide to use only the labeled data at unsupervised training.
            batch_size: Integer for the size of each generated batch
            n_sample_per_patient: Integer representing the number of samples per patient if sampling is set to 'pp'.
            semi: Boolean to decide whether or not to generate a pretext task label corresponding to (patient_id, time)
            temporal_window: Size of window to sample from  if sampling set to 'pp'. -1 means no window.
            pad_last_batch: Bolean Whether or not to pad last batch if below batch_size.
            task: String for the name of the task associated in the h5 file.
        """
        # We set sampling config
        self.window_size = window_size
        self.pad = pad
        self.padding_size = padding_size
        self.sequence_size = window_size + padding_size
        self.offsets = offsets
        self.shuffle = shuffle
        self.sampling_method = sampling_method
        self.batch_size = batch_size
        self.n_sample_per_patient = n_sample_per_patient
        self.semi = semi
        self.temporal_window = (temporal_window * step_size) // 2
        self.pad_last_batch = pad_last_batch

        if (2 * self.temporal_window < self.n_sample_per_patient) and self.temporal_window > 0:
            raise Exception('Temporal window is to small for the number of sample per patient')

        if self.offsets is None:
            self.max_offset = 0
            self.num_offset = 0
        else:
            self.max_offset = np.max(self.offsets) * step_size
            self.num_offset = len(self.offsets)

        self.step_size = step_size
        self.data_h5 = h5py.File(data_path, "r")
        self.lookup_table = self.data_h5['data']
        if 'tasks' in list(self.data_h5['labels'].attrs.keys()):
            self.labels_name = self.data_h5['labels'].attrs['tasks']
        else:
            self.labels_name = None

        if on_RAM:
            if nb_variables < 0:
                print('Loading look up table on RAM')
                self.lookup_table = {'train': self.data_h5['data']['train'][:], 'test': self.data_h5['data']['test'][:],
                                     'val': self.data_h5['data']['val'][:]}
            else:
                print('Loading look up table on RAM witht the first {} variables based on varations in time'.format(
                    nb_variables))
                self.lookup_table = {'train': self.data_h5['data']['train'][:, :nb_variables],
                                     'test': self.data_h5['data']['test'][:, :nb_variables],
                                     'val': self.data_h5['data']['val'][:, :nb_variables]}

        self.patient_start_stop_ids = {'train': self.data_h5['patient_windows']['train'][:],
                                       'test': self.data_h5['patient_windows']['test'][:],
                                       'val': self.data_h5['patient_windows']['val'][:]}

        # Processing on the label part for the evaluating part
        if task is None:
            assert ((len(self.data_h5['labels']['train'].shape) == 1) or (self.data_h5['labels'].shape[-1] == 1))
            self.label_idx = 0
        else:
            self.label_idx = np.where(self.labels_name == task)[0]
        self.labels = self.data_h5['labels']
        if on_RAM:
            if len(self.data_h5['labels']['train'].shape) > 1:
                self.labels = {'train': np.reshape(self.data_h5['labels']['train'][:, self.label_idx], (-1,)),
                               'test': np.reshape(self.data_h5['labels']['test'][:, self.label_idx], (-1,)),
                               'val': np.reshape(self.data_h5['labels']['val'][:, self.label_idx], (-1,))}
            else:
                self.labels = {'train': self.data_h5['labels']['train'][:],
                               'test': self.data_h5['labels']['test'][:],
                               'val': self.data_h5['labels']['val'][:]}
        self.valid_indexes_labels = {'train': list(np.argwhere(~np.isnan(self.labels['train'][:])).T[0]),
                                     'test': list(np.argwhere(~np.isnan(self.labels['test'][:])).T[0]),
                                     'val': list(np.argwhere(~np.isnan(self.labels['val'][:])).T[0])}

        self.num_labels = {'train': len(self.valid_indexes_labels['train']),
                           'test': len(self.valid_indexes_labels['test']),
                           'val': len(self.valid_indexes_labels['val'])}

        self.valid_indexes_us = {'train': [], 'test': [], 'val': []}
        self.valid_indexes_us_per_patient = {'train': {}, 'test': {}, 'val': {}}
        self.valid_indexes_labels_per_patient = {'train': {}, 'test': {}, 'val': {}}
        self.relative_indexes = {'train': [], 'test': [], 'val': []}
        self.idx_to_id = {'train': [], 'test': [], 'val': []}

        for split in ['train', 'test', 'val']:
            for start, stop, id_ in self.patient_start_stop_ids[split]:
                unique_id = id_
                while unique_id in self.valid_indexes_us_per_patient[split].keys():
                    unique_id += 1000000
                valid_idx = list(range(start, stop + 1 - self.max_offset))
                self.valid_indexes_us[split] += valid_idx
                if len(valid_idx) > self.n_sample_per_patient:
                    self.valid_indexes_us_per_patient[split][unique_id] = np.array(valid_idx)
                correct_labels = list(np.argwhere(~np.isnan(self.labels[split][start:stop + 1])).T[0] + start)
                if correct_labels:
                    self.valid_indexes_labels_per_patient[split][unique_id] = np.array(correct_labels)
                self.relative_indexes[split] += list(range(0, stop + 1 - start))
                self.idx_to_id[split] += list(np.ones((stop + 1 - start,)) * unique_id)

        if only_labeled:
            self.valid_indexes_us = self.valid_indexes_labels
            self.valid_indexes_us_per_patient = self.valid_indexes_labels_per_patient

        for split in ['train', 'test', 'val']:
            self.relative_indexes[split] = np.array(self.relative_indexes[split])
            self.valid_indexes_us[split] = np.array(self.valid_indexes_us[split])
            self.valid_indexes_labels[split] = np.array(self.valid_indexes_labels[split])
            self.idx_to_id[split] = np.array(self.idx_to_id[split])

        self.num_samples = {'train': len(self.valid_indexes_us['train']), 'test': len(self.valid_indexes_us['test']),
                            'val': len(self.valid_indexes_us['val'])}

        # Weights balance fort labels:
        if balance:
            if isinstance(balance, bool):
                pos_num_val = np.nansum(self.labels['val'][:]) / self.num_labels['val']
                pos_num_test = np.nansum(self.labels['test'][:]) / self.num_labels['test']
                pos_num_train = np.nansum(self.labels['train'][:]) / self.num_labels['train']
                self.labels_weights = {'train': [0.5 / (1 - pos_num_train), 0.5 / pos_num_train],
                                       'test': [0.5 / (1 - pos_num_test), 0.5 / pos_num_test],
                                       'val': [0.5 / (1 - pos_num_val), 0.5 / pos_num_val]}
            else:
                self.labels_weights = {'train': balance,
                                       'test': balance,
                                       'val': balance}


        else:
            self.labels_weights = {'train': [1.0, 1.0],
                                   'test': [1.0, 1.0],
                                   'val': [1.0, 1.0]}
        # Iterate counters
        self.current_index_training = {'train': 0, 'test': 0, 'val': 0}
        self.current_index_evaluating = {'train': 0, 'test': 0, 'val': 0}
        self.bias_init = [0.0, np.nansum(self.labels['train'][:]) / self.num_labels['train'] / (
                1 - np.nansum(self.labels['train'][:]) / self.num_labels['train'])]

        # Per patient variables
        self.valid_id_eval = {'train': np.array(list(self.valid_indexes_labels_per_patient['train'].keys())),
                              'test': np.array(list(self.valid_indexes_labels_per_patient['test'].keys())),
                              'val': np.array(list(self.valid_indexes_labels_per_patient['val'].keys()))}

        self.valid_id_us = {'train': np.array(list(self.valid_indexes_us_per_patient['train'].keys())),
                            'test': np.array(list(self.valid_indexes_us_per_patient['test'].keys())),
                            'val': np.array(list(self.valid_indexes_us_per_patient['val'].keys()))}

        self.num_patient_eval = {'train': len(self.valid_id_eval['train']),
                                 'test': len(self.valid_id_eval['test']),
                                 'val': len(self.valid_id_eval['val'])}

        self.num_patient_us = {'train': len(self.valid_id_us['train']),
                               'test': len(self.valid_id_us['test']),
                               'val': len(self.valid_id_us['val'])}

    def sort_indexes(self):
        """
        Function to resort the indexes at evaluation if needed.
        Returns:

        """
        for split in ['train', 'test', 'val']:
            self.valid_indexes_labels[split] = np.sort(self.valid_indexes_labels[split])
            self.valid_indexes_us[split] = np.sort(self.valid_indexes_us[split])

    def sample(self, random_state, split='train', training=False, idx_patient=None):
        """Function to sample from the data split of choice.

        This methods is further wrapped into a generator to build a tf.data.Dataset

        Args:
            random_state: np.random.RandomState instance to for the idx choice.
            split: String representing split to sample from, either 'train' or 'test'.
            training: Boolean to determine if we are training the representation.
            idx_patient: (Optional) array  to sample particular samples from given  indexes.

        Returns:
            A batch from the desired distribution as tuple of numpy arrays if semi or not training. Otherwise a array.
        """

        if training:
            if idx_patient is None:
                if self.sampling_method == 'ps':
                    idx_patient = random_state.randint(self.num_samples[split], size=(self.batch_size,))
                    state_idx = self.valid_indexes_us[split][idx_patient]

                elif self.sampling_method == 'pp':
                    idx_id = random_state.choice(np.arange(self.num_patient_us[split]), replace=False,
                                                 size=(int(np.ceil(self.batch_size / self.n_sample_per_patient)),))
                    idx_id = self.valid_id_us[split][idx_id]
                    possible_states = [self.valid_indexes_us_per_patient[split].get(id_) for id_ in idx_id]

                    if self.temporal_window > 0:

                        anchor_limits_states = []
                        for possible_state in possible_states:
                            if len(possible_state) > 2 * self.temporal_window + 1:
                                center = random_state.randint(self.temporal_window, len(
                                    possible_state) - self.temporal_window, size=(1,))
                            else:
                                center = len(possible_state) // 2

                            anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                        anchor_limits_states = np.array(anchor_limits_states)

                        lower_neighbor = np.max(
                            [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                        higher_neighbor = np.min(
                            [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)

                        state_idx = np.concatenate(
                            [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,))
                             for
                             low, high in
                             zip(lower_neighbor, higher_neighbor)])
                    else:
                        state_idx = np.concatenate(
                            [random_state.choice(pos_state, replace=False, size=(self.n_sample_per_patient,))
                             for pos_state in possible_states])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

                else:
                    raise Exception("Sampling methods is either 'pp' or 'ps'")
            else:
                if self.sampling_method == 'ps':
                    state_idx = np.reshape(self.valid_indexes_us[split][idx_patient], (-1,))

                elif self.sampling_method == 'pp':
                    idx_id = self.valid_id_us[split][idx_patient]
                    possible_states = [self.valid_indexes_us_per_patient[split].get(id_) for id_ in idx_id]
                    if self.temporal_window > 0:

                        anchor_limits_states = []
                        for possible_state in possible_states:
                            if len(possible_state) > 2 * self.temporal_window + 1:
                                center = random_state.randint(self.temporal_window, len(
                                    possible_state) - self.temporal_window, size=(1,))
                            else:
                                center = len(possible_state) // 2

                            anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                        anchor_limits_states = np.array(anchor_limits_states)

                        lower_neighbor = np.max(
                            [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                        higher_neighbor = np.min(
                            [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)

                        if np.any(lower_neighbor >= higher_neighbor):
                            print(lower_neighbor, higher_neighbor)
                        state_idx = np.concatenate(
                            [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,))
                             for
                             low, high in
                             zip(lower_neighbor, higher_neighbor)])
                    else:
                        state_idx = np.concatenate(
                            [random_state.choice(pos_state, replace=False, size=(self.n_sample_per_patient,)) for
                             pos_state in possible_states])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

            y = np.stack(
                [self.labels[split][state_idx], self.idx_to_id[split][state_idx], state_idx]).T


        else:
            if idx_patient is None:
                if self.sampling_method == 'ps':
                    idx_patient = random_state.randint(self.num_labels[split], size=(self.batch_size,))
                    state_idx = self.valid_indexes_labels[split][idx_patient]

                elif self.sampling_method == 'pp':
                    idx_id = random_state.choice(np.arange(self.num_patient_eval[split]), replace=False,
                                                 size=(int(np.ceil(self.batch_size / self.n_sample_per_patient)),))
                    idx_id = self.valid_id_eval[split][idx_id]
                    possible_states = [self.valid_indexes_labels_per_patient[split].get(id_) for id_ in idx_id]

                    anchor_limits_states = []
                    for possible_state in possible_states:
                        if len(possible_state) > 2 * self.temporal_window + 1:
                            center = random_state.randint(self.temporal_window, len(
                                possible_state) - self.temporal_window, size=(1,))
                        else:
                            center = len(possible_state) // 2

                        anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                    anchor_limits_states = np.array(anchor_limits_states)

                    lower_neighbor = np.max(
                        [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                    higher_neighbor = np.min(
                        [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)
                    state_idx = np.concatenate(
                        [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,)) for
                         low, high in
                         zip(lower_neighbor, higher_neighbor)])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

                else:
                    raise Exception("Sampling methods is either 'pp' or 'ps'")
            else:
                if self.sampling_method == 'ps':
                    state_idx = np.reshape(self.valid_indexes_labels[split][idx_patient], (-1,))

                elif self.sampling_method == 'pp':
                    idx_id = self.valid_id_eval[split][idx_patient]

                    possible_states = [self.valid_indexes_labels_per_patient[split].get(id_) for id_ in idx_id]

                    anchor_limits_states = []
                    for possible_state in possible_states:
                        if len(possible_state) > 2 * self.temporal_window + 1:
                            center = random_state.randint(self.temporal_window, len(
                                possible_state) - self.temporal_window, size=(1,))
                        else:
                            center = len(possible_state) // 2

                        anchor_limits_states.append([possible_state[center], possible_state[0], possible_state[-1]])
                    anchor_limits_states = np.array(anchor_limits_states)
                    lower_neighbor = np.max(
                        [anchor_limits_states[:, 1], anchor_limits_states[:, 0] - self.temporal_window], axis=0)
                    higher_neighbor = np.min(
                        [anchor_limits_states[:, 2], anchor_limits_states[:, 0] + self.temporal_window], axis=0)
                    state_idx = np.concatenate(
                        [random_state.choice(np.arange(low, high), replace=False, size=(self.n_sample_per_patient,)) for
                         low, high in
                         zip(lower_neighbor, higher_neighbor)])
                    random_state.shuffle(state_idx)
                    state_idx = state_idx[:self.batch_size]

            y = self.labels[split][state_idx]

        relative_idx = self.relative_indexes[split][state_idx]
        start = state_idx - relative_idx
        relative_start = relative_idx % self.step_size + start
        relative_idx = relative_idx // self.step_size
        windowing_inputs = np.stack([state_idx, relative_start, relative_idx], axis=-1)
        X = np.array(list(
            map(lambda x: self.windowing(x[0], x[1], x[2], self.offsets, self.window_size, self.padding_size, split,
                                         self.step_size,
                                         self.pad), windowing_inputs)))
        if self.semi or (not training):
            return X, y
        else:
            return X

    def iterate(self, random_state, split='train', training=False):
        """Function to iterate over the data split of choice.

        This methods is further wrapped into a generator to build a tf.data.Dataset.
        It wraps around the sample method.

        Args:
            random_state: np.random.RandomState instance to for the idx choice.
            split: String representing split to sample from, either 'train', 'test' or 'val'.
            training: Boolean to determine if we are training the representation.

        Returns:
            A sample corresponding to the current_index from the desired distribution as tuple of numpy arrays.
        """
        if training:
            if self.sampling_method == 'ps':
                if (self.current_index_training[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_indexes_us[split])

                next_idx = list(
                    range(self.current_index_training[split], self.current_index_training[split] + self.batch_size))
                self.current_index_training[split] += self.batch_size

                if self.current_index_training[split] >= self.num_samples[split]:
                    n_exceeding_samples = self.current_index_training[split] - self.num_samples[split]
                    assert n_exceeding_samples <= self.batch_size
                    if not self.pad_last_batch:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_training[split] = 0
            else:

                if (self.current_index_training[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_id_us[split])
                n_patient = int(np.ceil(self.batch_size / self.n_sample_per_patient))
                next_idx = list(
                    range(self.current_index_training[split], self.current_index_training[split] + n_patient))
                self.current_index_training[split] += n_patient

                if self.current_index_training[split] >= self.num_patient_us[split]:
                    n_exceeding_samples = self.current_index_training[split] - self.num_patient_us[split]
                    assert n_exceeding_samples <= n_patient
                    if not self.pad_last_batch:
                        next_idx = next_idx[:n_patient - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:n_patient - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_training[split] = 0

            sample = self.sample(random_state, split, training, idx_patient=next_idx)

            return sample

        else:
            if self.sampling_method == 'ps':
                if (self.current_index_evaluating[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_indexes_labels[split])

                next_idx = list(
                    range(self.current_index_evaluating[split], self.current_index_evaluating[split] + self.batch_size))
                self.current_index_evaluating[split] += self.batch_size

                if self.current_index_evaluating[split] >= self.num_labels[split]:
                    n_exceeding_samples = self.current_index_evaluating[split] - self.num_labels[split]
                    assert n_exceeding_samples <= self.batch_size
                    if not self.pad_last_batch:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:self.batch_size - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_evaluating[split] = 0
            else:
                if (self.current_index_evaluating[split] == 0) and self.shuffle:
                    random_state.shuffle(self.valid_id_eval[split])
                n_patient = int(np.ceil(self.batch_size / self.n_sample_per_patient))
                next_idx = list(
                    range(self.current_index_evaluating[split], self.current_index_evaluating[split] + n_patient))
                self.current_index_evaluating[split] += n_patient

                if self.current_index_evaluating[split] >= self.num_patient_eval[split]:
                    n_exceeding_samples = self.current_index_evaluating[split] - self.num_patient_eval[split]
                    assert n_exceeding_samples <= n_patient
                    if not self.pad_last_batch:
                        next_idx = next_idx[:n_patient - n_exceeding_samples]
                    else:
                        next_idx = next_idx[:n_patient - n_exceeding_samples] + list(range(n_exceeding_samples))
                    self.current_index_evaluating[split] = 0

            sample = self.sample(random_state, split, training, idx_patient=next_idx)
            return sample

    def windowing(self, state_idx, relative_start, relative_idx, offsets=None, window_size=200, padding_size=0,
                  split='train', step_size=1, pad=True):
        """Function extracting a window correponding a state index.

        Args:
            state_idx: Integer representing the state index in the lookup-table.
            relative_start: Integer representing the start of the patient stay.
            relative_idx: Integer representing the (end of the window)/(index of the state) in the stay.
            offsets: Integer setting the number of off-sets for the window if needed.
            window_size: Integer of the size of the window to sample.
            padding_size: Integer for size of the padding to add to all windows.
            split: String with the split of the data to take the window from.
            step_size: Integer with relative step size for smaller resolution windows.
            pad: Boolean for whether or to pad shorter windows to window_size.

        Returns:
                A window  as numpy array of shape (num_features, padding_size + len(offsets) + window_size).
        """

        if relative_idx + 1 < window_size:
            seq = self.lookup_table[split][relative_start: state_idx + 1: step_size]
            if pad:
                seq = np.concatenate(
                    [np.zeros((padding_size + window_size - relative_idx - 1, self.lookup_table[split].shape[-1])),
                     seq], axis=0)
        else:
            seq = self.lookup_table[split][state_idx + step_size - window_size * step_size:state_idx + 1:step_size]
            if padding_size != 0 and pad:
                seq = np.concatenate([-np.ones((padding_size, self.lookup_table[split].shape[-1])), seq])

        if offsets is not None:
            off_sets = [self.lookup_table[split][state_idx + k * step_size] for k in offsets]
            seq = np.concatenate([seq, off_sets], axis=0)

        return seq

    def sample_shape(self, training=False, split='train'):
        if not training:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1]), (
                None,)
        elif self.semi:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1]), (
                None, 3)
        else:
            return (None, self.padding_size + self.window_size, self.lookup_table[split].shape[-1])

    def sample_type(self, training=False):
        if not training or self.semi:
            return tf.float32, tf.float32
        else:
            return tf.float32
