import numpy as np
import tensorflow as tf


def tf_data_set_from_ground_truth_data(ground_truth_data, random_seed, training=False, split='train',
                                       gen_type='sample'):
    """Generate a tf.data.DataSet from ground_truth data."""

    if gen_type == 'sample':
        def generator():
            random_state = np.random.RandomState(random_seed)
            while True:
                yield ground_truth_data.sample(random_state, split, training)

    elif gen_type == 'iterate':
        random_state = np.random.RandomState(random_seed)

        def generator():
            if training:
                if ground_truth_data.sampling_method == 'ps':
                    num_samples = ground_truth_data.num_samples[split]
                else:
                    num_samples = ground_truth_data.num_patient_us[split]

                counter = 0
            else:
                if ground_truth_data.sampling_method == 'ps':
                    num_samples = ground_truth_data.num_labels[split]
                else:
                    num_samples = ground_truth_data.num_patient_eval[split]

                counter = 0

            if ground_truth_data.sampling_method == 'ps':
                num_samples = np.ceil(num_samples / ground_truth_data.batch_size)

            elif ground_truth_data.sampling_method == 'pp':
                num_samples = np.ceil(
                    num_samples / np.ceil(ground_truth_data.batch_size / ground_truth_data.n_sample_per_patient))

            while counter < num_samples:
                counter += 1
                yield ground_truth_data.iterate(random_state, split, training)
    else:
        raise DataError("gen_type should be in ['sample', iterate']")

    output_shapes = ground_truth_data.sample_shape(training)
    output_types = ground_truth_data.sample_type(training)

    ds = tf.data.Dataset.from_generator(generator, output_types, output_shapes=output_shapes)
    if (gen_type == 'iterate' and split == 'train') or (gen_type == 'iterate' and training):
        ds = ds.repeat()
    return ds


def correct_structured_h5(h5_file):
    if not ['data', 'labels', 'patient_windows'] == list(h5_file.keys()):
        raise DataError("Not correct subgroups  ['data', 'labels', 'patient_windows'] ")
    if not ['test', 'train', 'val'] == list(h5_file['data'].keys()):
        raise DataError("Not correct subgroups in data ")
    if not ['test', 'train', 'val'] == list(h5_file['labels'].keys()):
        raise DataError("Not correct subgroups in labels ")


class DataError(Exception):
    pass
