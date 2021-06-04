import gin
import tensorflow as tf

Adam_keras = gin.external_configurable(tf.keras.optimizers.Adam, name='Adam')
SGD_keras = gin.external_configurable(tf.keras.optimizers.SGD, name='SGD')


@gin.configurable('LinearWarmupCosineDecay')
class LinearWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a Cosine decay schedule."""

    def __init__(self,
                 initial_learning_rate,
                 warmup_steps,
                 warm_learning_rate,
                 decay_steps,
                 alpha=None,
                 name=None):

        super(LinearWarmupCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.warm_learning_rate = warm_learning_rate
        self.decay_steps = decay_steps
        self.rate = (warm_learning_rate - initial_learning_rate) / warmup_steps
        if not alpha:
            self.alpha = initial_learning_rate / warm_learning_rate
        else:
            self.alpha = alpha
        self.name = name
        self.cosine_decay = tf.keras.experimental.CosineDecay(self.warm_learning_rate, self.decay_steps, self.alpha)

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, step):
        with tf.name_scope(self.name or "LinearWarmupCosineDecay") as name:
            if step < self.warmup_steps:
                initial_learning_rate = tf.convert_to_tensor(
                    self.initial_learning_rate, name="initial_learning_rate")
                dtype = initial_learning_rate.dtype
                rate = tf.cast(self.rate, dtype)

                global_step_recomp = tf.cast(step, dtype)
                p = rate * global_step_recomp + initial_learning_rate
                return p
            else:
                return self.cosine_decay(step - self.warmup_steps)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "rate": self.rate,
            "name": self.name
        }
