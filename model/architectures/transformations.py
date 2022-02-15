import gin
import tensorflow as tf


@gin.configurable('spatial_dropout')
def spatial_dropout(temporal_sequence, tf_random_generator, rate=0.1):
    """
    Drop entire measurements from the sequences.
    Args:
        temporal_sequence: Input 3D tensor.
        tf_random_generator: Common tf.random.Generator.
        rate: Fraction of the measurement to drop.
    Returns:
        Transformed tensor with same shape.
    """
    noise_shape = list(temporal_sequence.shape)
    noise_shape[1] = 1
    noise_1d_mask = tf.cast(tf_random_generator.uniform(noise_shape) > rate, dtype=tf.float32)
    noise_mask = tf.tile(noise_1d_mask, (1, temporal_sequence.shape[1], 1))

    return temporal_sequence * noise_mask


@gin.configurable('normal_bias')
def add_normal_bias(temporal_sequence, tf_random_generator, std=0.1):
    """Add independent Gaussian noise to each channel.
    
    Args:
        temporal_sequence: Input 3D tensor.
        tf_random_generator: Common tf.random.Generator.
        std: Float with standard deviation of the noise to add.

    Returns:
        Transformed tensor with same shape.
    """
    padding_mask = tf.cast((temporal_sequence != 0.0), dtype=tf.float32)  # To not add noise to padding
    bias_shape = tf.concat([temporal_sequence.shape[:-2], [1], [temporal_sequence.shape[-1]]], axis=-1)
    noise = tf_random_generator.truncated_normal(bias_shape, stddev=std)

    return temporal_sequence + padding_mask * noise


@gin.configurable('history_crop')
def history_crop(temporal_sequence, tf_random_generator, padding_value=0.0, p=0.5, min_history=0.5):
    """Crops the history of temporal sequences without altering the last state. 
    
    Args:
        temporal_sequence: Input 3D tensor.
        tf_random_generator: Common tf.random.Generator.
        padding_value: Float with the value to use to pad the removed history.
        p: Float with the probability to apply this transformation to any element of the batch.
        min_history: Float with the minimal fraction of history to preserve. 
    Returns:
            Transformed tensor with same shape.
    """
    NUM_SEQ, SEQ_LENGTH, N_MEAS = temporal_sequence.shape
    padding_mask = tf.reduce_mean(temporal_sequence, axis=-1) != padding_value
    tmp_indices = tf.where(padding_mask)
    padding_size = tf.cast(tf.math.segment_min(tmp_indices[:, 1], tmp_indices[:, 0]) / SEQ_LENGTH, dtype=tf.float32)
    padding_size = tf.expand_dims(padding_size, axis=-1)
    #padding_size = tf.reduce_mean(padding_mask, axis=-1, keepdims=True)


    boxes_start = tf_random_generator.uniform(shape=(NUM_SEQ, 1))
    boxes_start = tf.cast((boxes_start * (1 - padding_size) * (1 - min_history) + padding_size) * SEQ_LENGTH, dtype=tf.int32)
    boxes_start = tf.expand_dims(boxes_start, axis=-1)

    # Only crop from left side
    cropping_mask = tf.tile(tf.reshape(tf.range(SEQ_LENGTH, dtype=tf.int32), (1, SEQ_LENGTH, 1)),
                            (NUM_SEQ, 1, 1)) >= boxes_start

    random_selection = tf.reshape(tf.cast(tf_random_generator.uniform(shape=(NUM_SEQ,)) < p, dtype=tf.float32),
                                  (-1, 1, 1))
    return temporal_sequence * (1 - random_selection) + temporal_sequence * tf.cast(cropping_mask,
                                                                                    dtype=tf.float32) * random_selection


@gin.configurable('history_cutout')
def history_cutout(temporal_sequence, tf_random_generator, size=4, p=0.5):
    """Mask out a small portion of the temporal sequences without altering the last state. 
    
    Args:
        temporal_sequence: Input 3D tensor.
        tf_random_generator: Common tf.random.Generator.
        size: Integer with the size of the cutouts.
        p: Float with the probability to apply this transformation to any element of the batch.

    Returns:
            Transformed tensor with same shape.
    """
    size = tf.cast(size, dtype=tf.int32)
    NUM_SEQ, SEQ_LENGTH, N_MEAS = temporal_sequence.shape
    relative_size = tf.cast(size / (SEQ_LENGTH), dtype=tf.float32)

    # We ensure last state is never masked by enforcing the last possible crop start to be (SEQ_LENGTH - size - 1)
    boxes_start = tf_random_generator.uniform(shape=(NUM_SEQ, 1), minval=0, maxval=(1 - relative_size) - 1 / SEQ_LENGTH)
    boxes_start = tf.cast(boxes_start * SEQ_LENGTH, dtype=tf.int32)
    boxes_start = tf.expand_dims(boxes_start, axis=-1)

    cropping_mask_pre = tf.cast((tf.tile(tf.reshape(tf.range(SEQ_LENGTH, dtype=tf.int32), (1, SEQ_LENGTH, 1)),
                                         (NUM_SEQ, 1, 1)) < boxes_start), dtype=tf.float32)
    cropping_mask_post = tf.cast((tf.tile(tf.reshape(tf.range(SEQ_LENGTH, dtype=tf.int32), (1, SEQ_LENGTH, 1)),
                                          (NUM_SEQ, 1, 1)) >= boxes_start + size), dtype=tf.float32)
    cropping_mask = cropping_mask_pre + cropping_mask_post

    random_selection = tf.reshape(tf.cast(tf_random_generator.uniform(shape=(NUM_SEQ,)) < p, dtype=tf.float32),
                                  (-1, 1, 1))
    return temporal_sequence * (1 - random_selection) + temporal_sequence * tf.cast(cropping_mask,
                                                                                    dtype=tf.float32) * random_selection


@gin.configurable('composition')
def compose_transformation(x, tf_random_generator, functions):
    """Wrapper around functions to apply them sequentially.
    
    """
    transformed = [x]
    for f in functions:
        transformed.append(f(transformed[-1], tf_random_generator))
    return transformed[-1]
