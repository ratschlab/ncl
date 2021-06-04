import gin
import tensorflow as tf


# NCE Contrastive Losses

@gin.configurable("momentum_NT_X")
def Momentum_NT_X(features, temperature=1.0):
    q, queue = features
    N = tf.shape(q)[0]
    C = tf.shape(queue)[1]
    k = queue[:N]  # The first N elememts of the queue correspond to the positive views.

    # Compute positives
    l_pos = tf.matmul(tf.reshape(q, [N, 1, C]), tf.reshape(k, [N, C, 1]))
    l_pos = tf.reshape(l_pos, [N, 1])

    # Compute normalization
    logits = tf.matmul(q, tf.transpose(queue))
    labels = tf.range(N, dtype=tf.int32)  # Meta-labels for the contrastive accuracy

    expectations_marginal_per_sample = tf.reduce_logsumexp(logits / temperature, axis=1)
    joint_expectation = tf.reduce_sum(l_pos, axis=1) / temperature

    X_ent_per_sample = joint_expectation - expectations_marginal_per_sample
    loss = - tf.reduce_mean(X_ent_per_sample, axis=0)

    # Computing the contrastive accuracy
    preds = tf.transpose(tf.argmax(tf.squeeze(logits), axis=1, output_type=tf.int32))
    correct_preds = (preds == labels)
    accuracy = tf.reduce_mean(tf.cast(tf.expand_dims(correct_preds, axis=0), tf.float32), axis=1)[0]
    return loss, accuracy


@gin.configurable("momentum_neighbors_NT_X")
def Momentum_Neighbors_NT_X(features, n_labels_queues, temperature=1.0, alpha=0.5, neigh_func=gin.REQUIRED):
    """
    Neighborhood contrastive loss used with Momentum queue as : https://arxiv.org/abs/1911.05722
    Args:
        features: Tuple of the form (projections, queue)
        n_labels_queues: Tensor with matching labels to the momentum queue.
        temperature: Float with temperature scaling parameter.
        alpha: Float with trade-off parameter between inter and intra-neighborhood learning.
        neigh_func: function used to define neigborhoods.

    Returns:
        loss: Global loss term.
        aggregation_loss: Neighbors aggregation term.
        disc_loss: Neighbors discrimation term.
        accuracy: Contrastive accuracy.
    """

    q, queue = features
    N = tf.shape(q)[0]
    C = tf.shape(queue)[1]
    k = queue[:N]
    n_l = n_labels_queues[:N]

    # Compute true positives from view
    local_pos = tf.matmul(tf.reshape(q, [N, 1, C]), tf.reshape(k, [N, C, 1]))
    local_pos = tf.reshape(local_pos, [N, 1]) / temperature
    local_labels = tf.range(N, dtype=tf.int32)
    joint_expectation = tf.reduce_sum(local_pos, axis=1)

    # Compute true normalization
    logits = tf.matmul(q, tf.transpose(queue)) / temperature
    expectations_marginal_per_sample = tf.reduce_logsumexp(logits, axis=1)

    # Neighborhood terms
    # Neighborhood positives for aggregation
    neighbors_mask = neigh_func(n_l, n_labels_queues)
    number_neigh = tf.reduce_sum(neighbors_mask, axis=1)
    neighbors_expectation = tf.reduce_sum(logits * neighbors_mask, axis=1) / number_neigh
    aggregation_loss = tf.reduce_mean(expectations_marginal_per_sample - neighbors_expectation)

    # Neighborhood negatives for discrimination
    expectations_neighborhood_per_sample = tf.math.log(
        tf.reduce_sum(tf.math.exp(logits) * neighbors_mask, axis=1))
    n_X_ent_per_sample = expectations_neighborhood_per_sample - joint_expectation
    disc_loss = tf.reduce_mean(n_X_ent_per_sample)

    loss = alpha * aggregation_loss + (1.0 - alpha) * disc_loss

    # Computing the contrastive accuracy
    preds = tf.transpose(tf.argmax(tf.squeeze(logits), axis=1, output_type=tf.int32))
    correct_preds = (preds == local_labels)
    accuracy = tf.reduce_mean(tf.cast(tf.expand_dims(correct_preds, axis=0), tf.float32))

    return loss, aggregation_loss, disc_loss, accuracy


# Neighborhood Functions

@gin.configurable('get_neighbors_mask_temporal')
def get_neighbors_mask_temporal(samples, queue, threshold=8):
    """Neighborhood function to aggregate samples from same same patient with w hours.

    Args:
        samples: Tensor with samples labels (N, L).
        queue:  Tensor with queue labels (K, L).
        threshold: Integer with max temporal distance between neighbors.

    Returns:
        Mask with the shape (N, K)
    """
    N = samples.shape[0]
    K = queue.shape[0]
    a_n, a_t = tf.split(tf.tile(tf.reshape(samples, (N, 1, -1)), (1, K, 1)), 2, axis=-1)
    b_n, b_t = tf.split(tf.tile(tf.reshape(queue, (1, K, -1)), (N, 1, 1)), 2, axis=-1)
    t_n = a_n - b_n
    t_t = tf.abs(a_t - b_t)
    c = tf.math.logical_and((t_n == 0), (t_t <= threshold))
    return tf.cast(c[:, :, 0], dtype=tf.float32)


@gin.configurable('get_neighbors_dt_multiclass')
def get_neighbors_dt_label_multiclass(samples, queue):
    """Neighborhood function to aggregate samples with same downstrean task label.

    Args:
        samples: Tensor with samples labels (N, L).
        queue:  Tensor with queue labels (K, L).

    Returns:
        Mask with the shape (N, K)
    """
    N = samples.shape[0]
    K = queue.shape[0]
    a_n = tf.tile(tf.reshape(samples, (N, 1, -1)), (1, K, 1))
    b_n = tf.tile(tf.reshape(queue, (1, K, -1)), (N, 1, 1))
    t_n = a_n - b_n
    c = tf.cast((t_n == 0), dtype=tf.float32)[:, :, 0] + tf.linalg.diag(tf.ones(N), num_rows=N, num_cols=K)
    return tf.cast(c >= 1, dtype=tf.float32)


@gin.configurable('get_neighbors_nascl_label_multiclass')
def get_neighbors_nascl_label_multiclass(samples, queue, threshold=8, joint='intersection'):
    """Neighborhood function to aggregate samples from same patient with w hours and/or with same DT label.

    Args:
        samples: Tensor with samples labels (N, L).
        queue:  Tensor with queue labels (K, L).
        threshold: Integer with max temporal distance between neighbors.
        joint: String either 'intersection' or 'union' to determine how to aggregate the two neighborhoods.

    Returns:
        Mask with the shape (N, K)
    """
    samples_dt, samples_patient = tf.split(samples, [1, 2], axis=-1)
    queue_dt, queue_patient = tf.split(queue, [1, 2], axis=-1)
    neigh_dt = get_neighbors_dt_label_multiclass(samples_dt, queue_dt)
    neigh_patient = get_neighbors_mask_temporal(samples_patient, queue_patient, threshold=threshold)
    if joint == 'intersection':
        return neigh_dt * neigh_patient
    elif joint == 'union':
        return tf.cast((neigh_patient + neigh_dt >= 1), dtype=tf.float32)
