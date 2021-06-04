import gin
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


@gin.configurable('addon_classification')
class AddOnClassification(tf.Module):
    def __init__(self, representation_model=gin.REQUIRED, classifier=gin.REQUIRED, loss_fn=gin.REQUIRED,
                 optimizer=gin.REQUIRED, patience=10, end_to_end=False, stopping_metric='loss',
                 grad_clip=None, profile=False):
        """Wrapper around the classifier we build on top of the representation model.

        Args:
            representation_model: tf.Module corresponding to the representation model we train at previous step.
            classifier: tf.keras.Model to use on top of representation model.
            loss_fn: tf.keras loss function.
            optimizer: tf.keras.optimizer.
            patience: Integer setting the patience for the early stopping criterion.
            end_to_end: Boolean to decide whether or not to train the entire model (encoder + add_on classifier).
            stopping_metric : String telling which metric to track for the stopping criterion.
            grad_clip: Value to clip gradient if needed. None means no gradient clipping.
            profile: Boolean indicating whether or not we want to profile the training to evaluate efficiency.
        """
        super(AddOnClassification, self).__init__()

        self.representation_fn = representation_model.encoder
        self.classifier = classifier

        # Reduction is set to None for correct handling of distributed training.
        self.loss = loss_fn(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = optimizer
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.auc = tf.keras.metrics.AUC(num_thresholds=1000)
        self.auprc = tf.keras.metrics.AUC(curve='PR', num_thresholds=1000)
        self.patience = patience
        self.stopping_metric = stopping_metric
        self.grad_clip = grad_clip
        self.profile = profile
        self.tf_random_generator = None

        if end_to_end:
            self.representation_fn.trainable = True

    def set_tf_random_generator(self, generator):
        self.tf_random_generator = generator

    def step_fn(self, inputs, training, num_replica, loss_weights=[0.5, 0.5], augmentations=[]):
        """
        Step function encapsulated in model_fn.
        """
        if len(inputs) == 2:
            sequences, labels = inputs
        else:
            sequences = inputs[:2]
            labels = inputs[-1]
        # Handling of unbalanced training set for the loss computation.
        sample_weight = tf.cast(labels, dtype=tf.float32) * loss_weights[1] - (
                tf.cast(labels, dtype=tf.float32) - 1) * loss_weights[0]

        if training:
            if augmentations:
                for aug_func in augmentations:
                    sequences = aug_func(sequences, self.tf_random_generator)

            with tf.GradientTape() as tape_supervised:
                embeddings = self.representation_fn(sequences, training=training)
                predictions = self.classifier(embeddings, training=training)
                if isinstance(self.loss, tfa.losses.SigmoidFocalCrossEntropy):
                    loss_supervised = tf.reduce_mean(
                        self.loss(tf.expand_dims(labels, axis=-1), predictions[:, 1:], sample_weight=sample_weight)) * (
                                              1 / num_replica)
                else:
                    loss_supervised = tf.reduce_mean(self.loss(labels, predictions, sample_weight=sample_weight)) * (
                            1 / num_replica)

            if self.representation_fn.trainable == True:
                variables = tape_supervised.watched_variables()
            else:
                variables = self.classifier.trainable_variables
            grads = tape_supervised.gradient(loss_supervised, variables)

            if self.grad_clip:
                grads, global_norm = tf.clip_by_global_norm(grads, self.grad_clip)

            self.optimizer.apply_gradients(zip(grads, variables))
            self.accuracy.update_state(labels, predictions)
            self.auc.update_state(labels, predictions[:, 1])
            self.auprc.update_state(labels, predictions[:, 1])
            return loss_supervised

        else:
            embeddings = self.representation_fn(sequences, training=training)
            predictions = self.classifier(embeddings, training=training)
            if isinstance(self.loss, tfa.losses.SigmoidFocalCrossEntropy):
                loss_supervised = tf.reduce_mean(
                    self.loss(tf.expand_dims(labels, axis=-1), predictions[:, 1:], sample_weight=sample_weight)) * (
                                          1 / num_replica)
            else:
                loss_supervised = tf.reduce_mean(self.loss(labels, predictions, sample_weight=sample_weight)) * (
                        1 / num_replica)
            self.accuracy.update_state(labels, predictions)
            self.auc.update_state(labels, predictions[:, 1])
            self.auprc.update_state(labels, predictions[:, 1])
            return loss_supervised

    @tf.function
    def model_fn(self, inputs, training, strategy, loss_weights=(0.5, 0.5), augmentations=[]):
        """Model function for the downstream task training.

        Args:
            inputs: Inputs corresponding to (self.representation_fn.inputs, label).
            training: Boolean to decide whether to train model for a step.
            strategy: (Optional) tf.distribute.Strategy object in case of distributed training.
            loss_weights: (Optional) weights to apply to loss functions in the case of an imbalanced dataset.
            augmentations: (Optional) List of augmentations we want to apply to each input.
        Returns:
            loss: Tensor corresponding to the loss over a step.

        """
        if strategy is not None:
            loss_per_sub_batch = strategy.experimental_run_v2(self.step_fn,
                                                              args=(inputs, training, strategy.num_replicas_in_sync,
                                                                    loss_weights, augmentations))
            loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, loss_per_sub_batch, axis=None)
        else:
            loss = self.step_fn(inputs, training, num_replica=1, loss_weights=loss_weights, augmentations=augmentations)

        return loss

    def train(self, data_iterator, training_steps, loss_weights=(0.5, 0.5), strategy=None, summary_writer=None,
              checkpoint_manager=None, validation_config=None, augmentations=[]):

        """Custom train loop for the add on classifier.

        Args:
            data_iterator: An iterator yielding batch samples of the data.
            training_steps: Integer representing the number of iterations.
            loss_weights: weights to apply to loss functions in the case of an imbalanced dataset.
            strategy: (Optional) tf.distribute.Strategy object in case of distributed training.
            summary_writer: (Optional) tf.summary.Writer for tensorboard logging.
            checkpoint_manager: (Optional) tf.train.CheckpointManager for model saving.
            validation_config: (Optional) Dictionary with the necessary params for validation.
            augmentations: List of augmentations functions to apply to input.
        """

        if checkpoint_manager is None:
            print('There is no model saving')

        # Setting up validation phase is a config is provided
        if validation_config is not None:
            validation_frequency = validation_config['frequency']
        else:
            validation_frequency = training_steps + 1

        best_val_metric = tf.cast(-np.inf, tf.float32)
        plateau = 0
        train_loss = 0.0
        ts_dir = '/'.join(checkpoint_manager.directory.split('/')[:-1] + ['tensorboard'])
        for step in range(training_steps):
            if step == 1000 and self.profile:
                print('Open profiling')
                tf.profiler.experimental.start(ts_dir)

            batch = data_iterator.next()

            with summary_writer.as_default():
                train_loss += self.model_fn(batch,
                                            training=True,
                                            strategy=strategy,
                                            loss_weights=tf.cast(loss_weights, dtype=tf.float32),
                                            augmentations=augmentations)

            if (step + 1) % validation_frequency == 0 and step != 0:
                if summary_writer:
                    with summary_writer.as_default():
                        tf.summary.scalar("train_loss", train_loss / validation_frequency,
                                          step=self.optimizer.iterations)
                        tf.summary.scalar("train_auc", self.auc.result(), step=self.optimizer.iterations)
                        tf.summary.scalar("train_auprc", self.auprc.result(), step=self.optimizer.iterations)
                        tf.summary.scalar("train_accuracy", self.accuracy.result(), step=self.optimizer.iterations)

                train_loss = 0.0
                self.auc.reset_states()
                self.auprc.reset_states()
                self.accuracy.reset_states()

                val_metrics = self.validate(validation_config['data_iterator'],
                                            loss_weights=loss_weights, strategy=strategy,
                                            summary_writer=summary_writer)

                if checkpoint_manager and (val_metrics[self.stopping_metric] > best_val_metric or self.patience <= 0):
                    best_val_metric = val_metrics[self.stopping_metric]
                    print('Saving model at step {}'.format(step))
                    checkpoint_manager.save()
                    plateau = 0
                else:
                    plateau += 1
            if plateau >= self.patience and self.patience > 0:
                print('Loss dit not improve for {} testing step'.format(self.patience))
                break

            if step == 1005 and self.profile:
                print('Close profiling')
                tf.profiler.experimental.stop()

    def validate(self, data_iterator, loss_weights=(0.5, 0.5), strategy=None, summary_writer=None):
        if summary_writer is None:
            print('There is no summary writer')
        loss = 0.0
        num_batch = 0.0
        for batch in data_iterator:
            loss += self.model_fn(batch, training=False, strategy=strategy,
                                  loss_weights=tf.cast(loss_weights, dtype=tf.float32))
            num_batch += 1.0
        metrics = {'loss': - loss / num_batch, 'auroc': self.auc.result(), 'auprc': self.auprc.result(),
                   'acc': self.accuracy.result()}

        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar("val_loss", - metrics['loss'], step=self.optimizer.iterations)
                tf.summary.scalar("val_auc", metrics['auroc'], step=self.optimizer.iterations)
                tf.summary.scalar("val_auprc", metrics['auprc'], step=self.optimizer.iterations)
                tf.summary.scalar("val_accuracy", metrics['acc'], step=self.optimizer.iterations)

        self.auc.reset_states()
        self.accuracy.reset_states()
        self.auprc.reset_states()
        return metrics

    def compute_metrics(self, data_iterator):
        self.auc.reset_states()
        self.accuracy.reset_states()
        self.auprc.reset_states()

        for batch in data_iterator:
            if len(batch) == 2:
                data, labels = batch
            else:
                data = batch[:2]
                labels = batch[-1]
            predictions = self.predict(data)
            self.accuracy.update_state(labels, predictions)
            self.auc.update_state(labels, predictions[:, 1])
            self.auprc.update_state(labels, predictions[:, 1])

        results = {'accuracy': self.accuracy.result(), 'auroc': self.auc.result(), 'auprc': self.auprc.result(),
                   'tp': self.auprc.true_positives, 'tn': self.auprc.true_negatives,
                   'fn': self.auprc.false_negatives, 'fp': self.auprc.false_positives}
        tn = self.auprc.true_negatives
        tp = self.auprc.true_positives
        fp = self.auprc.false_positives
        fn = self.auprc.false_negatives
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        fig = plt.figure()
        plt.plot(R, P)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('2-class Precision-Recall curve: AUPRC={0:0.2f}'.format(results['auprc']))
        return results, fig

    def predict(self, data):
        embeddings = self.representation_fn(data, training=False)
        predictions = self.classifier(embeddings, training=False)
        return predictions


@gin.configurable('addon_binned_classification')
class AddOnBinnedClassification(tf.Module):
    def __init__(self, representation_model=gin.REQUIRED, classifier=gin.REQUIRED, loss_fn=gin.REQUIRED,
                 optimizer=gin.REQUIRED, patience=10, end_to_end=False, stopping_metric='loss',
                 grad_clip=None, profile=False, bins=None):
        """Wrapper around the classifier we build on top of the representation model for a binned classification.

        Args:
            representation_model: tf.Module corresponding to the representation model we train at previous step.
            classifier: tf.keras.Model to use on top of representation model.
            loss_fn: tf.keras loss function.
            optimizer: tf.keras.optimizer.
            patience: Integer setting the patience for the early stopping criterion.
            end_to_end: Boolean to decide whether or not to train the entire model (encoder + add_on classifier).
            stopping_metric : String telling which metric to track for the stopping criterion.
            grad_clip: Value to clip gradient if needed. None means no gradient clipping.
            profile: Boolean indicating whether or not we want to profile the training to evaluate efficiency.
            bins: List of bins separator to build the classes on.
        """
        super(AddOnBinnedClassification, self).__init__()
        representation_model.encoder.int_op = []
        self.representation_fn = representation_model.encoder
        self.classifier = classifier
        # Reduction is set to None for correct handling of distributed training.
        self.loss = loss_fn(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = optimizer
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.metrics = {'acc': self.accuracy}

        if bins:
            self.bins = bins
            self.n_bins = len(bins) + 2
            self.kappa = tfa.metrics.CohenKappa(num_classes=self.n_bins, sparse_labels=True, weightage='linear')
            self.metrics['kappa'] = self.kappa

        self.patience = patience
        self.stopping_metric = stopping_metric
        self.grad_clip = grad_clip
        self.profile = profile
        self.tf_random_generator = None
        if end_to_end:
            self.representation_fn.trainable = True

    def set_tf_random_generator(self, generator):
        self.tf_random_generator = generator

    def step_fn(self, inputs, training, num_replica, augmentations=[]):
        """
        Step function encapsulated in model_fn.
        """
        if len(inputs) == 2:
            sequences, labels = inputs
        else:
            sequences = inputs[:2]
            labels = inputs[-1]
        labels = tf.expand_dims(labels, axis=-1)
        binned_labels = tf.raw_ops.Bucketize(input=labels, boundaries=self.bins)
        if training:
            if augmentations:
                for aug_func in augmentations:
                    sequences = aug_func(sequences, self.tf_random_generator)

            with tf.GradientTape() as tape_supervised:
                embeddings = self.representation_fn(sequences, training=training)
                predictions = self.classifier(embeddings, training=training)
                loss_supervised = tf.reduce_mean(self.loss(binned_labels, predictions)) * (
                        1 / num_replica)

            if self.representation_fn.trainable == True:
                variables = tape_supervised.watched_variables()
            else:
                variables = self.classifier.trainable_variables
            grads = tape_supervised.gradient(loss_supervised, variables)

            if self.grad_clip:
                grads, global_norm = tf.clip_by_global_norm(grads, self.grad_clip)

            self.optimizer.apply_gradients(zip(grads, variables))

        else:
            embeddings = self.representation_fn(sequences, training=training)
            predictions = self.classifier(embeddings, training=training)
            loss_supervised = tf.reduce_mean(self.loss(binned_labels, predictions)) * (
                    1 / num_replica)
        binned_pred = tf.math.argmax(predictions, axis=-1)

        for name, metric in self.metrics.items():
            if name == 'kappa':
                metric.update_state(binned_labels[:, 0], binned_pred)
            else:
                metric.update_state(binned_labels, predictions)
        return loss_supervised

    @tf.function
    def model_fn(self, inputs, training, strategy, augmentations=[]):
        """Model function for the downstream task training.

        Args:
            inputs: Inputs corresponding to (self.representation_fn.inputs, label).
            training: Boolean to decide whether to train model for a step.
            strategy: (Optional) tf.distribute.Strategy object in case of distributed training.
            augmentations: (Optional) List of augmentations we want to apply to each input.
        Returns:
            loss: Tensor corresponding to the loss over a step.
        """

        if strategy is not None:
            loss_per_sub_batch = strategy.experimental_run_v2(self.step_fn,
                                                              args=(inputs, training, strategy.num_replicas_in_sync,
                                                                    augmentations))
            loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, loss_per_sub_batch, axis=None)
        else:
            loss = self.step_fn(inputs, training, num_replica=1, augmentations=augmentations)

        return loss

    def train(self, data_iterator, training_steps, loss_weights=(0.5, 0.5), strategy=None, summary_writer=None,
              checkpoint_manager=None, validation_config=None, augmentations=[]):

        """Custom train loop for the add on classifier.

        Args:
            data_iterator: An iterator yielding batch samples of the data.
            training_steps: Integer representing the number of iterations.
            loss_weights: weights to apply to loss functions in the case of an imbalanced dataset.
            strategy: (Optional) tf.distribute.Strategy object in case of distributed training.
            summary_writer: (Optional) tf.summary.Writer for tensorboard logging.
            checkpoint_manager: (Optional) tf.train.CheckpointManager for model saving.
            validation_config: (Optional) Dictionary with the necessary params for validation.
            augmentations: List of augmentations functions to apply to input.
        """
        if checkpoint_manager is None:
            print('There is no model saving')

        # Setting up validation phase if a config is provided
        if validation_config is not None:
            validation_frequency = validation_config['frequency']
        else:
            validation_frequency = training_steps + 1

        best_val_metric = tf.cast(-np.inf, tf.float32)
        plateau = 0
        train_loss = 0.0
        ts_dir = '/'.join(checkpoint_manager.directory.split('/')[:-1] + ['tensorboard'])
        for step in range(training_steps):
            if step == 1000 and self.profile:
                print('Open profiling')
                tf.profiler.experimental.start(ts_dir)

            batch = data_iterator.next()

            with summary_writer.as_default():
                train_loss += self.model_fn(batch, training=True, strategy=strategy,
                                            augmentations=augmentations)

            if (step + 1) % validation_frequency == 0 and step != 0:
                if summary_writer:
                    with summary_writer.as_default():
                        tf.summary.scalar("train_loss", train_loss / validation_frequency,
                                          step=self.optimizer.iterations)
                        for name, metric in self.metrics.items():
                            tf.summary.scalar("train_" + name, metric.result(), step=self.optimizer.iterations)
                            metric.reset_states()
                train_loss = 0.0

                val_metrics = self.validate(validation_config['data_iterator'],
                                            strategy=strategy,
                                            summary_writer=summary_writer)

                if checkpoint_manager and (val_metrics[self.stopping_metric] > best_val_metric or self.patience <= 0):
                    best_val_metric = val_metrics[self.stopping_metric]
                    print('Saving model at step {}'.format(step))
                    checkpoint_manager.save()
                    plateau = 0
                else:
                    plateau += 1

            if plateau >= self.patience and self.patience > 0:
                print('Loss dit not improve for {} testing step'.format(self.patience))
                break
            if step == 1005 and self.profile:
                print('Close profiling')
                tf.profiler.experimental.stop()

    def validate(self, data_iterator, strategy=None, summary_writer=None):
        if summary_writer is None:
            print('There is no summary writer')
        loss = 0.0
        num_batch = 0.0
        for batch in data_iterator:
            loss += self.model_fn(batch, training=False, strategy=strategy)
            num_batch += 1.0
        metrics = {'loss': - loss / num_batch}
        for name, metric in self.metrics.items():
            metrics[name] = metric.result()
            metric.reset_states()

        if summary_writer:
            with summary_writer.as_default():
                tf.summary.scalar("val_loss", - metrics['loss'], step=self.optimizer.iterations)

                for name, metric in metrics.items():
                    if name != 'loss':
                        tf.summary.scalar("val_" + name, metric, step=self.optimizer.iterations)

        return metrics

    def compute_metrics(self, data_iterator):
        for name, metric in self.metrics.items():
            metric.reset_states()

        for batch in data_iterator:
            if len(batch) == 2:
                data, labels = batch
            else:
                data = batch[:2]
                labels = batch[-1]
            labels = tf.expand_dims(labels, axis=-1)
            predictions = self.predict(data)
            labels = tf.expand_dims(labels, axis=-1)
            binned_labels = tf.raw_ops.Bucketize(input=labels, boundaries=self.bins)
            binned_pred = tf.math.argmax(predictions, axis=-1)

            for name, metric in self.metrics.items():
                if name == 'kappa':
                    metric.update_state(binned_labels[:, 0], binned_pred)
                else:
                    metric.update_state(binned_labels, predictions)

        results = {name: metric.result() for name, metric in self.metrics.items()}
        fig = None
        return results, fig

    def predict(self, data):
        embeddings = self.representation_fn(data, training=False)
        predictions = self.classifier(embeddings, training=False)
        return predictions
