import os
import pickle
import random
import gin
import numpy as np
import tensorflow as tf

from data import utils
from eval.architectures.utils import load_post_eval_model, physionet2019_utility
from eval.down_stream_tasks.add_on_classification import AddOnClassification, AddOnBinnedClassification

def eval_with_gin(strategy_name=None,
                  model_dir=None,
                  overwrite=False,
                  gin_config_files=None,
                  gin_bindings=None,
                  seed=1234):
    """Evaluate a representation based on the provided gin configuration.

    This function will set the provided gin bindings, call the evaluate_rep() function
    and clear the gin config. Please see evaluate_rep() for required gin bindings.

    Args:
        strategy_name: String indicating the distribution strategy,
                       if None, the training is on a single gpu.
        model_dir: String with path to directory where model output should be saved.
        overwrite: Boolean indicating whether to overwrite output directory.
        gin_config_files: List of gin config files to load.
        gin_bindings: List of gin bindings to use.
        seed: Integer corresponding to the common seed used for any random operation.
    """

    # Setting the seeds before gin parsing
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
    if not isinstance(gin_config_files, list):
        gin_config_files = [gin_config_files]
    if not isinstance(gin_bindings, list):
        gin_bindings = [gin_bindings]
    if strategy_name == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    if strategy:
        with strategy.scope():
            gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
            evaluate_rep(model_dir, overwrite=overwrite, strategy=strategy, random_seed=seed,
                         tf_random_generator=tf_random_generator)
        gin.clear_config()
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
        eval_model(model_dir)
    else:
        gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
        evaluate_rep(model_dir, overwrite=overwrite, random_seed=seed, tf_random_generator=tf_random_generator)
    gin.clear_config()


@gin.configurable('eval_task')
def evaluate_rep(task_dir,
                 overwrite=False,
                 strategy=None,
                 random_seed=None,
                 task=gin.REQUIRED,
                 training_steps=gin.REQUIRED,
                 training_epochs=None,
                 validation_config=None,
                 dataset=gin.REQUIRED,
                 gen_type='iterate',
                 augmentations=[],
                 tf_random_generator=None,
                 compute_utility=False):
    """Train the downstream task classifier to evaluate the representation.

    At the end of the training the operating config and last checkpoint are saved to
    model_dir folder.

    Args:
        task_dir: String with path to directory where task training output should be saved.
        overwrite: Boolean indicating whether to overwrite output directory.
        strategy: tf.distribute.Strategy passed for in the case of distributed training.
        random_seed: Integer with the common random seed used for training.
        task: tf.Module containing the method's model to evaluate the representation.
        training_steps: Integer with total number of training steps to do.
        training_epochs: (Optional) Integer with total number of training epochs to do, overwrites training_steps.
        validation_config: Dict with the config for validation params.
        dataset: Object from data module to transform into a tf.data.Dataset.
        gen_type: String deciding whether to sample randomly from data or iterate over data.
        augmentations: List of augmentations functions for training.
        tf_random_generator: tf.random.generator built in eval_with_gin() and used for all tf operations.
        compute_utility: Boolean to compute or not utility. Relevant only to physionet2019.
    """
    if tf.io.gfile.isdir(task_dir):
        if overwrite:
            tf.io.gfile.rmtree(task_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    # If we evaluate the representation by adding some layer on top of the encoder network
    writer = tf.summary.create_file_writer(os.path.join(task_dir, 'tensorboard'))
    checkpoint_model = tf.train.Checkpoint(model=task)
    manager = tf.train.CheckpointManager(checkpoint_model, directory=os.path.join(task_dir, 'tf_checkpoints'),
                                         max_to_keep=1)

    with open(os.path.join(task_dir, 'eval_config.gin'), 'w') as f:
        f.write(gin.operative_config_str())

    # Setting up data iterators
    train_loss_weights = dataset.labels_weights['train']
    val_loss_weights = dataset.labels_weights['val']
    iter_data_train = make_eval_dataset_iterator(dataset, random_seed, strategy, split='train',
                                                 gen_type=gen_type)
    if validation_config is not None:
        validation_config['data_iterator'] = make_eval_dataset_iterator(dataset, random_seed, strategy,
                                                                        split='val', gen_type='iterate')
        validation_config['loss_weights'] = val_loss_weights
        if validation_config['frequency'] == -1:
            # We set validation to one per epoch.
            val_per_epoch = dataset.num_labels['train'] // dataset.batch_size + 1
            validation_config['frequency'] = val_per_epoch

    if training_epochs:
        training_steps = training_epochs * (dataset.num_labels['train'] // dataset.batch_size + 1)

    # Training
    task.set_tf_random_generator(tf_random_generator)
    task.train(data_iterator=iter_data_train, training_steps=training_steps, strategy=strategy,
               summary_writer=writer, checkpoint_manager=manager, loss_weights=train_loss_weights,
               validation_config=validation_config, augmentations=augmentations)

    print('Training is done !')
    # For memory sake
    iter_data_train = None

    # Computing metrics on trained model
    if strategy is None:
        dataset.sort_indexes()
        dataset.shuffle = False
        checkpoint_model.restore(manager.latest_checkpoint)
        dataset.current_index_evaluating['val'] = 0
        eval_dataset = make_eval_dataset_iterator(dataset, random_seed, strategy=None,
                                                  split='val', gen_type='iterate')

        val_metrics, fig_val = task.compute_metrics(data_iterator=eval_dataset)

        if compute_utility:
            val_patient_windows = dataset.patient_start_stop_ids['val']
            encoder = task.representation_fn
            classifier = task.classifier
            y_true = []
            y_pred = []
            for start, stop, id_ in val_patient_windows:
                idxs = np.arange(start, stop + 1)
                inputs, labels = dataset.sample(None, split='val', training=False, idx_patient=idxs)
                embeddings = encoder(tf.cast(inputs, dtype=tf.float32), training=False)
                preds = classifier(embeddings, training=False)[:, 1]
                y_pred.append(preds)
                y_true.append(labels)
            val_utility = physionet2019_utility(y_true, y_pred)
            val_metrics['utility'] = val_utility
            val_metrics['y_true'] = y_true
            val_metrics['y_pred'] = y_pred

        if fig_val is not None:
            fig_val.savefig(os.path.join(task_dir, 'val_prc.png'))

        dataset.current_index_evaluating['test'] = 0
        eval_dataset = make_eval_dataset_iterator(dataset, random_seed, strategy=None,
                                                  split='test', gen_type='iterate')
        metrics, fig_test = task.compute_metrics(data_iterator=eval_dataset)

        if compute_utility:
            test_patient_windows = dataset.patient_start_stop_ids['test']
            encoder = task.representation_fn
            classifier = task.classifier
            y_true = []
            y_pred = []
            for start, stop, id_ in test_patient_windows:
                idxs = np.arange(start, stop + 1)
                inputs, labels = dataset.sample(None, split='test', training=False, idx_patient=idxs)
                embeddings = encoder(tf.cast(inputs, dtype=tf.float32), training=False)
                preds = classifier(embeddings, training=False)[:, 1]
                y_pred.append(preds)
                y_true.append(labels)
            test_utility = physionet2019_utility(y_true, y_pred)
            metrics['utility'] = test_utility
            metrics['y_true'] = y_true
            metrics['y_pred'] = y_pred

        if fig_test is not None:
            fig_test.savefig(os.path.join(task_dir, 'test_prc.png'))

    if strategy is None:
        with open(os.path.join(task_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        with open(os.path.join(task_dir, 'val_metrics.pkl'), 'wb') as f:
            pickle.dump(val_metrics, f)

    # Saving the config for reproducibility
    with open(os.path.join(task_dir, 'eval_config.gin'), 'w') as f:
        f.write(gin.operative_config_str())


def eval_model(model_dir):
    dataset = gin.query_parameter('eval_task.dataset').scoped_configurable_fn()
    dataset.batch_size = 64
    eval_test_dataset = make_eval_dataset_iterator(dataset, 1111, strategy=None,
                                                   split='test', gen_type='iterate')
    model = load_post_eval_model(model_dir)
    metrics, fig_test = model.compute_metrics(data_iterator=eval_test_dataset)
    with open(os.path.join(model_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    fig_test.savefig(os.path.join(model_dir, 'test_prc.png'))
    eval_test_dataset = None
    eval_val_dataset = make_eval_dataset_iterator(dataset, 1111, strategy=None,
                                                  split='val', gen_type='iterate')
    metrics, fig_val = model.compute_metrics(data_iterator=eval_val_dataset)
    with open(os.path.join(model_dir, 'val_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    fig_val.savefig(os.path.join(model_dir, 'val_prc.png'))


def make_eval_dataset_iterator(ground_truth_data, seed, strategy=None, split='train', gen_type='sample'):
    """TF 2.3 custom loop eval compatible input function."""
    tf_dataset = utils.tf_data_set_from_ground_truth_data(ground_truth_data, seed, training=False, split=split,
                                                          gen_type=gen_type)

    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if strategy is not None:
        tf_dataset = strategy.experimental_distribute_dataset(tf_dataset)

    if split == 'train':
        return iter(tf_dataset)
    else:
        return tf_dataset
