import gin
import tensorflow as tf


@gin.configurable('momentum_semi_supervised')
class MomentumSemiSupervised(tf.Module):
    def __init__(self, encoder=gin.REQUIRED, projector=gin.REQUIRED, momentum_encoder=gin.REQUIRED,
                 momentum_projector=gin.REQUIRED, projection_size=gin.REQUIRED,
                 queue_size=1024, momentum=0.99, loss=gin.REQUIRED, optimizer=gin.REQUIRED,
                 transformations=[], label_size=2, labels_bin=[]):
        """Common momentum contrastive pipeline used for all contrastive methods.

        Args:
            encoder: tf.keras.Model corresponding to the representation encoder we train.
            projector: tf.keras.Model corresponding to projector.
            momentum_encoder: tf.keras.Model corresponding to the momentum encoder we use to build queue.
            momentum_projector: tf.keras.Model corresponding to the momentum projector.
            projection_size: Size of projection used to pre-build queue.
            queue_size: Integer with the length of the queue.
            momentum: Float with momentum parameter.
            loss: Function computing a loss from the outputs of the encoder.
            optimizer: tf.keras.optimizer instance for the entire set up.
            transformations: List of transformation used to build views.
            label_size: Size of the neighbors labels.
            labels_bin: Bins for specific case of SCL in trained on LOS.
        """

        super(MomentumSemiSupervised, self).__init__()
        self.encoder = encoder
        self.momentum_encoder = momentum_encoder
        self.loss = loss
        self.optimizer = optimizer
        self.projection_size = projection_size
        self.queue_size = queue_size
        self.momentum = momentum
        self.embedding_queue = tf.math.l2_normalize(tf.random.uniform((self.queue_size, self.projection_size), -1, 1),
                                                    axis=1)
        if label_size > 1:
            self.labels_queue = tf.cast(tf.random.uniform((self.queue_size, label_size), 0, 100, dtype=tf.int32),
                                        dtype=tf.float32)
        else:
            self.labels_queue = tf.cast(tf.random.uniform((self.queue_size,), 50, 100, dtype=tf.int32),
                                        dtype=tf.float32)

        self.transformations = transformations
        self.projector = projector
        self.momentum_projector = momentum_projector
        self.tf_random_generator = None
        self.labels_bin = labels_bin

    def set_random_generator(self, tf_random_generator):
        self.tf_random_generator = tf_random_generator

    @tf.function
    def model_fn(self, inputs, training, strategy, queue, n_queue):
        """Model function for the self supervised training of the encoder.

        Args:
            inputs: Inputs corresponding to self.encoder.inputs.
            training: Boolean to decide whether to train the representation encoder for a step.
            We choose the word 'training' for the flag to follow tf.keras flags.
            strategy: tf.distribute.Strategy object in case of distributed training.
            queue: Queue on prior momentum projections
            n_queue: Neighbor information queue matching the queue
        Returns:
            loss: L_NCL in the paper.
            aggregation_term: L_NA term in paper
            discrimination_term: L_ND in the paper.
            accuracy: Contrastive Accuracy
            new_keys: New keys to append to the queue.

        """

        def step_fn(inputs, n_labels, training, num_replica=1, queue=None, n_queue=None):
            """Step function for contrastive pipe.

            Args:
                inputs:  Tensor with batch of time-series
                n_labels: Tensor with neigbhorhood labels used to define neighborhood function.
                For SCL this is the downstream task labels.
                training: Training flag.
                num_replica: Integer with number of replica in case of distributed training.
                queue: Tensor with momentum queue.
                n_queue: Tensor with neighbors labels associated to the elements of the momentum queue.

            Returns:

            """
            x_1 = self.transformations[0](inputs, self.tf_random_generator)
            x_2 = self.transformations[1](inputs, self.tf_random_generator)

            mum_features_2 = self.momentum_encoder(x_2, training=training)
            new_keys_2 = tf.math.l2_normalize(self.momentum_projector(mum_features_2, training=training), axis=-1)
            mum_features_1 = self.momentum_encoder(x_1, training=training)
            new_keys_1 = tf.math.l2_normalize(self.momentum_projector(mum_features_1, training=training), axis=-1)

            # We take the second augmentations to update the queues

            concat_1 = tf.concat([new_keys_1, queue], axis=0)
            concat_2 = tf.concat([new_keys_2, queue], axis=0)

            labels_concat = tf.concat([n_labels, n_queue], axis=0)

            # We only watch the encoder for backprop
            with tf.GradientTape() as tape_unsupervised:
                embeddings_1 = self.encoder(x_1, training=training)
                embeddings_2 = self.encoder(x_2, training=training)
                projections_1 = tf.math.l2_normalize(self.projector(embeddings_1), axis=-1)
                projections_2 = tf.math.l2_normalize(self.projector(embeddings_2), axis=-1)

                loss_unsupervised_1, aggregation_term_1, local_term_1, acc_1 = self.loss([projections_1, concat_2],
                                                                                         labels_concat)
                loss_unsupervised_2, aggregation_term_2, local_term_2, acc_2 = self.loss([projections_2, concat_1],
                                                                                         labels_concat)
                loss_unsupervised = (loss_unsupervised_1 + loss_unsupervised_2) / 2
                aggregation_term = (aggregation_term_1 + aggregation_term_2) / 2
                local_term = (local_term_1 + local_term_2) / 2
                acc = (acc_1 + acc_2) / 2

                loss_unsupervised /= num_replica
                aggregation_term /= num_replica
                local_term /= num_replica
                acc /= num_replica

            if training:
                variables = self.encoder.trainable_variables + self.projector.trainable_variables

                grads = tape_unsupervised.gradient(loss_unsupervised, variables)
                self.optimizer.apply_gradients(zip(grads, variables))

            return loss_unsupervised, aggregation_term, local_term, acc, new_keys_1

        seq, n_labels = inputs

        if strategy is not None:
            num_replica = strategy.num_replicas_in_sync

            sub_batch_step_loss, sub_batch_step_agg, sub_batch_step_local, sub_batch_step_accuracy, sub_new_keys = strategy.experimental_run_v2(
                step_fn, args=(seq, n_labels, training, num_replica, queue, n_queue))

            loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, sub_batch_step_loss, axis=None)
            local_term = strategy.reduce(
                tf.distribute.ReduceOp.SUM, sub_batch_step_local, axis=None)
            aggregation_term = strategy.reduce(
                tf.distribute.ReduceOp.SUM, sub_batch_step_agg, axis=None)
            accuracy = strategy.reduce(
                tf.distribute.ReduceOp.SUM, sub_batch_step_accuracy, axis=None)

            if num_replica > 1:
                new_keys = tf.concat(sub_new_keys.values, axis=0)
                new_labels = tf.concat(n_labels.values, axis=0)
            else:
                new_keys = sub_new_keys
                new_labels = n_labels
        else:
            loss, aggregation_term, local_term, accuracy, new_keys = step_fn(seq, n_labels, training, queue=queue,
                                                                             n_queue=n_queue)
            new_labels = n_labels

        if training:
            pair_variables = zip(self.encoder.trainable_variables + self.projector.trainable_variables,
                                 self.momentum_encoder.trainable_variables + self.momentum_projector.trainable_variables)
            for encoder_var, m_encoder_var in pair_variables:
                m_encoder_var.assign(self.momentum * m_encoder_var + (1.0 - self.momentum) * encoder_var)

        return loss, aggregation_term, local_term, accuracy, new_keys, new_labels

    def train(self, data_iterator, training_steps, strategy=None, summary_writer=None,
              checkpoint_manager=None, monitoring_config=None):
        """Custom train loop for the self supervision method.

        Args:
            data_iterator: An iterator yielding batch samples of the data.
            training_steps: Integer representing the number of iterations.
            strategy: (Optional) tf.distribute.Strategy object in case of distributed training.
            summary_writer: (Optional) tf.summary.Writer for tensorboard logging.
            checkpoint_manager: (Optional) tf.train.CheckpointManager for model saving.
        """
        freq = training_steps // 10
        if summary_writer is None:
            print('There is no summary writer')
        if checkpoint_manager is None:
            print('There is no model saving')

        if monitoring_config is not None:
            monitoring_frequency = monitoring_config['frequency']
        else:
            monitoring_frequency = training_steps + 1

        with summary_writer.as_default():

            for step in range(training_steps):
                inputs = data_iterator.next()
                if self.labels_bin:
                    binned_labels = inputs[1]
                    if len(binned_labels.shape) > 1:

                        binned_labels = tf.concat((tf.cast(
                            tf.raw_ops.Bucketize(input=binned_labels[:, :1], boundaries=self.labels_bin),
                            dtype=tf.float32), binned_labels[:, 1:]), axis=1)
                    else:
                        binned_labels = tf.cast(tf.raw_ops.Bucketize(input=binned_labels, boundaries=self.labels_bin),
                                                dtype=tf.float32)
                    inputs = (inputs[0], binned_labels)

                loss, aggregation_term, local_term, local_acc, new_keys, new_n_labels = self.model_fn(inputs,
                                                                                                      training=True,
                                                                                                      strategy=strategy,
                                                                                                      queue=self.embedding_queue,
                                                                                                      n_queue=self.labels_queue)

                tf.summary.scalar("Negative Loss", - loss, step=self.optimizer.iterations)
                tf.summary.scalar('Negative Aggregation Loss', - aggregation_term,
                                  step=self.optimizer.iterations)
                tf.summary.scalar('Negative Local Loss', - local_term,
                                  step=self.optimizer.iterations)
                tf.summary.scalar('Contrastive Accuracy', local_acc,
                                  step=self.optimizer.iterations)

                new_queue = tf.concat([new_keys, self.embedding_queue], axis=0)
                if new_queue.shape[0] > self.queue_size:
                    new_queue = new_queue[:self.queue_size]
                self.embedding_queue = new_queue

                new_labels_queue = tf.concat([new_n_labels, self.labels_queue], axis=0)
                if new_labels_queue.shape[0] > self.queue_size:
                    new_labels_queue = new_labels_queue[:self.queue_size]
                self.labels_queue = new_labels_queue

                if step % monitoring_frequency == 0 and step != 0:
                    val_loss = self.monitoring(monitoring_config['data_iterator'], monitoring_config['steps'], strategy,
                                               summary_writer)

                if checkpoint_manager and step % freq == 0:
                    print('Saving mode at step {}'.format(step))
                    checkpoint_manager.save()

    def monitoring(self, data_iterator, steps, strategy=None, summary_writer=None):

        for step in range(steps):
            inputs = data_iterator.next()
            if self.labels_bin:
                binned_labels = inputs[1]
                if len(binned_labels.shape) > 1:

                    binned_labels = tf.concat((tf.cast(
                        tf.raw_ops.Bucketize(input=binned_labels[:, :1], boundaries=self.labels_bin), dtype=tf.float32),
                                               binned_labels[:, 1:]), axis=1)
                else:
                    binned_labels = tf.cast(tf.raw_ops.Bucketize(input=binned_labels, boundaries=self.labels_bin),
                                            dtype=tf.float32)
                inputs = (inputs[0], binned_labels)

            if step == 0:
                tot_loss, tot_aggregation_term, tot_local_term, tot_local_acc, new_keys, new_n_labels = self.model_fn(
                    inputs, training=False,
                    strategy=strategy, queue=self.embedding_queue, n_queue=self.labels_queue)

            else:
                loss, aggregation_term, local_term, local_acc, new_keys, new_n_labels = self.model_fn(inputs,
                                                                                                      training=False,
                                                                                                      strategy=strategy,
                                                                                                      queue=self.embedding_queue,
                                                                                                      n_queue=self.labels_queue)
                tot_loss += loss
                tot_aggregation_term += aggregation_term
                tot_local_term += local_term
                tot_local_acc += local_acc

        tot_loss /= steps
        tot_aggregation_term /= steps
        tot_local_term /= steps
        tot_local_acc /= steps

        tf.summary.scalar("Negative Loss Val", - tot_loss, step=self.optimizer.iterations)
        tf.summary.scalar('Negative Aggregation Loss Val', - tot_aggregation_term,
                          step=self.optimizer.iterations)
        tf.summary.scalar('Negative Local Loss Val', - tot_local_term,
                          step=self.optimizer.iterations)
        tf.summary.scalar('Contrastive Accuracy Val', tot_local_acc,
                          step=self.optimizer.iterations)
        return tot_loss
