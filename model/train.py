import os
import random

import gin
import numpy as np
import tensorflow as tf

from data import utils


def train_with_gin(strategy_name=None,
                   model_dir=None,
                   overwrite=False,
                   gin_config_files=None,
                   gin_bindings=None,
                   seed=1234):
    """Trains a model based on the provided gin configuration.

    This function will set the provided gin bindings, call the train() function
    and clear the gin config. Please see train() for required gin bindings.

    Args:
        strategy_name: String indicating the distribution strategy,
                       if None, the training is on a single gpu.
        model_dir: String with path to directory where model output should be saved.
        overwrite: Boolean indicating whether to overwrite output directory.
        gin_config_files: List of gin config files to load.
        gin_bindings: List of gin bindings to use.
        seed: Integer corresponding to the common seed used for any random operation.
    """

    # Setting the seed before gin parsing
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf_random_generator = tf.random.Generator.from_seed(seed)


    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []

    if strategy_name == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    if strategy:
        # We need to create generator outside of the strategy scope

        with strategy.scope():
            gin.parse_config_files_and_bindings([gin_config_files], gin_bindings)
            train(model_dir, overwrite=overwrite, strategy=strategy, random_seed=seed,
                  tf_random_generator=tf_random_generator)
    else:
        gin.parse_config_files_and_bindings([gin_config_files], gin_bindings)
        train(model_dir, overwrite=overwrite, random_seed=seed, tf_random_generator=tf_random_generator)
    gin.clear_config()


@gin.configurable('model')
def train(model_dir,
          overwrite=True,
          strategy=None,
          random_seed=None,
          tf_random_generator=None,
          model=gin.REQUIRED,
          training_steps_global=gin.REQUIRED,
          dataset=gin.REQUIRED,
          monitoring_config=None,
          gen_type='iterate'):
    """Trains the representation encoder.

    At the end of the training the operating config and last checkpoint are saved to
    model_dir folder.

    Args:
        model_dir: String with path to directory where model output should be saved.
        overwrite: Boolean indicating whether to overwrite output directory.
        strategy: tf.distribute.Strategy passed for in the case of distributed training.
        random_seed: Integer with random seed used for training.
        tf_random_generator: tf.random.Generator built in train_with_gin() and used for all tf operations.
        model: tf.Module containing the method's model to learn the representation.
        training_steps_global: Integer with total number of training steps to do.
        dataset: Object from data module to transform into a tf.data.Dataset.
        monitoring_config: Dict with config for monitoring the training.
        gen_type: Type of data generation from the dataset.
    """
    if tf.io.gfile.isdir(model_dir):
        if overwrite:
            tf.io.gfile.rmtree(model_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    model.set_random_generator(tf_random_generator)

    # Logging and saving set up
    writer = tf.summary.create_file_writer(os.path.join(model_dir, 'tensorboard'))
    checkpoint_model = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint_model, directory=os.path.join(model_dir, 'tf_checkpoints'), max_to_keep=None)

    # Saving the config for reproducibility
    with open(os.path.join(model_dir, 'train_config.gin'), 'w') as f:
        f.write(gin.operative_config_str())

    num_replica = 1
    if strategy is not None:
        num_replica = strategy.num_replicas_in_sync

    # We create a dataset per replica thus we need to divide the local batchsize
    dataset.batch_size = dataset.batch_size // num_replica

    # Building iterator from tf_dataset
    iter_data = make_dataset_iterator(dataset, random_seed, strategy, split='train', gen_type=gen_type)

    if monitoring_config is not None:
        monitoring_config['data_iterator'] = make_dataset_iterator(dataset, random_seed, strategy,
                                                                   split='val')
    # Training
    model.train(data_iterator=iter_data, training_steps=training_steps_global, strategy=strategy, summary_writer=writer,
                checkpoint_manager=manager, monitoring_config=monitoring_config)

    # Saving the config for reproducibility
    with open(os.path.join(model_dir, 'train_config.gin'), 'w') as f:
        f.write(gin.operative_config_str())


def make_dataset_iterator(ground_truth_data, seed, strategy=None, split='train', gen_type='iterate'):
    """TF 2.3 custom loop training compatible input fuction."""
    if strategy is None:
        tf_dataset = utils.tf_data_set_from_ground_truth_data(ground_truth_data, seed, training=True, split=split,
                                                              gen_type=gen_type)

        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    else:
        def dataset_fn(input_context):
            tf_dataset = utils.tf_data_set_from_ground_truth_data(ground_truth_data, seed, training=True, split=split,
                                                                  gen_type=gen_type)
            tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

            return tf_dataset

        tf_dataset = strategy.experimental_distribute_datasets_from_function(dataset_fn)

    return iter(tf_dataset)
