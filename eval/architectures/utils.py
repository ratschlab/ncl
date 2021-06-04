import os

import gin
import numpy as np
import tensorflow as tf


@gin.configurable('load_representation')
def load_unsupervised_model(model_dir, load_weights=True, ckpt=None):
    """Load the pretrained representation learned at the previous pipeline step.

    Args:
        model_dir: String with the path to the logging dir for the given model.
        load_weights: Boolean to load the weights of the model.
        ckpt: String indicating the name of the checkpoint to take. If None takes last.

    Returns:
        A trained tf.Module instance with a model_fn to corresponding to the
        trained encoder function.
    """
    checkpoint_path = os.path.join(model_dir, 'tf_checkpoints')
    unsupervised_model = gin.query_parameter('model.model').scoped_configurable_fn()
    if load_weights:
        checkpoint_model = tf.train.Checkpoint(model=unsupervised_model)
        manager = tf.train.CheckpointManager(checkpoint_model, directory=checkpoint_path, max_to_keep=3)
        if not manager.latest_checkpoint:
            raise Exception('No model to load at {}'.format(model_dir))

        if not ckpt:
            checkpoint_model.restore(manager.latest_checkpoint).expect_partial()
        else:
            checkpoint_model.restore(os.path.join(checkpoint_path, ckpt)).expect_partial()

    unsupervised_model.encoder.trainable = False

    return unsupervised_model


@gin.configurable('load_ft_representation')
def load_post_eval_model(model_dir):
    """Load the post evaluation model.

    Args:
        model_dir: String with the path to the logging dir for the given model

    Returns:
        A  tf.Module instance with a trained encoder and classifier.
    """
    current_config = gin.config.config_str()
    with open(os.path.join(model_dir, 'current_config.gin'), 'w') as f:
        f.write(current_config)
    with gin.unlock_config():
        gin.parse_config_files_and_bindings([os.path.join(model_dir, 'eval_config.gin')], [])
    checkpoint_path = os.path.join(model_dir, 'tf_checkpoints')
    ft_model = gin.query_parameter('eval_task.task').scoped_configurable_fn()
    checkpoint_model = tf.train.Checkpoint(model=ft_model)
    manager = tf.train.CheckpointManager(checkpoint_model, directory=checkpoint_path, max_to_keep=3)
    print(manager.latest_checkpoint)
    checkpoint_model.restore(manager.latest_checkpoint).expect_partial()
    with gin.unlock_config():
        gin.parse_config_files_and_bindings([os.path.join(model_dir, 'current_config.gin')], [])

    return ft_model


def build_representation(dataset, encoder, split, n_patient=500):
    """Build the representation of a split given a encoder.

    Args:
        dataset: Dataset instance from data.loader.
        encoder: tf.keras.Model that encode the input into the representation.
        split: String with the name of the split of interest in dataset
        n_patient: Integer with the number of patient to take. -1 means all.

    Returns:
        rep: Dict with the representation and label for each patient state.
        true_state: Dict with the true value of the input variable at each state.
    """

    dataset.sort_indexes()
    dataset.shuffle = False
    patient_windows = dataset.patient_start_stop_ids[split]
    rep = {}
    true_state = {}
    for start, stop, id_ in patient_windows[:n_patient]:
        idxs = np.arange(start, stop + 1)
        inputs, labels = dataset.sample(None, split=split, training=False, idx_patient=idxs)
        embeddings = encoder(tf.cast(inputs, dtype=tf.float32), training=False)
        rep[id_] = (embeddings, labels)
        true_state[id_] = inputs[:, -1, :]
    return rep, true_state


### METRICS from https://github.com/BorgwardtLab/Set_Functions_for_Time_Series/blob/master/seft/evaluation_metrics.py
def compute_prediction_utility(labels, predictions, dt_early=-12,
                               dt_optimal=-6, dt_late=3.0, max_u_tp=1,
                               min_u_fn=-2, u_fp=-0.05, u_tn=0,
                               check_errors=True):
    """Compute utility score of physionet 2019 challenge."""
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)


def physionet2019_utility(y_true, y_score):
    """Compute physionet 2019 Sepsis early detection utility.
    Code based on:

    Args:
        y_true: list from which each element is a the sequence of labels for a single patient.
        y_score: corresponding prediction to y_true.
    Returns:
    """
    dt_early = -12
    dt_optimal = -6
    dt_late = 3.0

    utilities = []
    best_utilities = []
    inaction_utilities = []

    for labels, observed_predictions in zip(y_true, y_score):
        observed_predictions = np.round(observed_predictions)
        num_rows = len(labels)
        best_predictions = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            pred_begin = int(max(0, t_sepsis + dt_early))
            pred_end = int(min(t_sepsis + dt_late + 1, num_rows))
            best_predictions[pred_begin:pred_end] = 1

        utilities.append(
            compute_prediction_utility(labels, observed_predictions))
        best_utilities.append(
            compute_prediction_utility(labels, best_predictions))
        inaction_utilities.append(
            compute_prediction_utility(labels, inaction_predictions))

    unnormalized_observed_utility = sum(utilities)
    unnormalized_best_utility = sum(best_utilities)
    unnormalized_inaction_utility = sum(inaction_utilities)
    normalized_observed_utility = (
            (unnormalized_observed_utility - unnormalized_inaction_utility)
            / (unnormalized_best_utility - unnormalized_inaction_utility)
    )
    return normalized_observed_utility
