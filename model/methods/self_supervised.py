import gin
import tensorflow as tf


@gin.configurable('auto_encoder')
class AutoEncoding(tf.Module):
    def __init__(self, encoder=gin.REQUIRED, decoder=gin.REQUIRED, optimizer=gin.REQUIRED, transformations=[],
                 forcast_horizon=None):

        """

        Args:
            encoder: tf.keras.Model corresponding to the representation encoder we train.
            decoder: tf.keras.Model corresponding to the representation decoder.
            optimizer: tf.keras.optimizer instance for the entire set up.
            transformations: List of transformations .
            forcast_horizon: Integer with horizon to predict in case of AE-forecast.
        """
        super(AutoEncoding, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = optimizer

        # We don't want to update the momentum encoder using backprop
        self.trainable_models = {'encoder': self.encoder, 'decoder': self.decoder}
        self.transformations = transformations
        self.tf_random_generator = None
        self.forcast_horizon = forcast_horizon

    def set_random_generator(self, tf_random_generator):
        self.tf_random_generator = tf_random_generator

    @tf.function
    def model_fn(self, inputs, training, strategy):
        """Model function for the self supervised training of the encoder.

        Args:
            inputs: Inputs corresponding to self.encoder.inputs.
            training: Boolean to decide whether to train the representation encoder for a step.
            We choose the word 'training' for the flag to follow tf.keras flags.
            strategy: tf.distribute.Strategy object in case of distributed training.

        Returns:
            loss: Tensor corresponding to the loss over a step.

        """

        def step_fn(inputs, training, n_rep=1):
            """Step function.

            Args:
                inputs: Same as model_fn
                training: Training flag
                n_rep: Number of replica. If no distribution set to 1 as defaults.

            Returns:
                MSE loss.
            """

            if self.forcast_horizon:
                seq_length = inputs.shape[1]
                seq, true_reconstrutions = tf.split(inputs, [seq_length - self.forcast_horizon, self.forcast_horizon],
                                                    axis=1)
            else:
                seq = inputs
                true_reconstrutions = inputs
            true_static_reconstructions, true_variable_reconstructions = tf.split(true_reconstrutions, (
            self.decoder.n_static, true_reconstrutions.shape[-1] - self.decoder.n_static), axis=-1)
            true_static_reconstructions = true_static_reconstructions[:,-1, :]

            if self.transformations:
                x_1 = self.transformations[0](seq, self.tf_random_generator)
            else:
                x_1 = seq

            with tf.GradientTape() as tape_unsupervised:
                embeddings = self.encoder(x_1, training=training)
                embeddings = tf.expand_dims(embeddings, axis=1)
                static_reconstructions, variable_reconstruction = self.decoder(embeddings, training=training)

                loss_unsupervised = self.loss(true_static_reconstructions, static_reconstructions) + self.loss(
                    true_variable_reconstructions, variable_reconstruction)

                loss_unsupervised /= n_rep

            if training:
                variables = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape_unsupervised.gradient(loss_unsupervised, variables)
                self.optimizer.apply_gradients(zip(grads, variables))
            return loss_unsupervised

        if strategy is not None:
            n_rep = strategy.num_replicas_in_sync
            sub_batch_step_loss = strategy.experimental_run_v2(
                step_fn, args=(inputs, training, n_rep))

            loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, sub_batch_step_loss, axis=None)
        else:
            loss = step_fn(inputs, training)

        return loss

    def train(self, data_iterator, training_steps, strategy=None, summary_writer=None,
              checkpoint_manager=None, monitoring_config=None):
        """Custom train loop for the auto encoding method.

        Args:
            data_iterator: An iterator yielding batch samples of the data.
            training_steps: Integer representing the number of iterations.
            strategy: (Optional) tf.distribute.Strategy object in case of distributed training.
            summary_writer: (Optional) tf.summary.Writer for tensorboard logging.
            checkpoint_manager: (Optional) tf.train.CheckpointManager for model saving.
            monitoring_config: Config for monitoring on validation set.
        """
        freq = training_steps // 10  # Saving 10 checkpoints of the model
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
                loss = self.model_fn(inputs, training=True, strategy=strategy)

                tf.summary.scalar("Negative Loss", - loss, step=self.optimizer.iterations)

                if step % monitoring_frequency == 0 and step != 0:

                    val_loss = self.monitoring(monitoring_config['data_iterator'], monitoring_config['steps'], strategy,
                                               summary_writer)

                if checkpoint_manager and step % freq == 0:
                    print('Saving mode at step {}'.format(step))
                    checkpoint_manager.save()

    def monitoring(self, data_iterator, steps, strategy=None, summary_writer=None):

        for step in range(steps):
            inputs = data_iterator.next()

            if step == 0:
                tot_loss = self.model_fn(inputs, training=False, strategy=strategy)
            else:
                loss = self.model_fn(inputs, training=False, strategy=strategy)
                tot_loss += loss
        tot_loss /= steps

        tf.summary.scalar("Negative Loss Val", - tot_loss, step=self.optimizer.iterations)
        return tot_loss
