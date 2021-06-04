import gin
import tensorflow as tf
from tcn import TCN


@gin.configurable("simple_TCN")
class simple_TCN(tf.keras.Model):
    """
    Sequence encoder based on TCN architectures from https://arxiv.org/pdf/1803.01271.pdf
    TCN block requires keras-tcn  from https://github.com/philipperemy/keras-tcn.
    """

    def __init__(self, kernel_size=gin.REQUIRED, n_stack=gin.REQUIRED, n_filter=gin.REQUIRED,
                 dilatations=(1, 2, 4, 8, 16, 32), do=0.0, static_do=0.0, activation='relu', use_skip=False,
                 use_LN=True,
                 l2_norm=True, n_static=gin.REQUIRED, embedding_size=None, input_shape=None):
        """Init function.

        Args:
            kernel_size: Size for the time axis of the kernel.
            n_stack: Integer for the number of residual block to stack.
            n_filter: Number of filters for the convolutions. For now it should be left at the input size.
            dilatations: List of integer corresponding to the succesives dilatations rates.
            do: Float representing dropout within TCN.
            static_do: Float for dropout to apply to static features.
            activation: String with name for activation used through out the model.
            use_skip: Boolean to whether or not to used skip layers as in a WaveNet style model.
            use_LN: Boolean to whether or not to use layer norm in the TCN blocks.
            l2_norm: Boolean to whether to whether or not project the representations to unit sphere.
            n_static: Integer with number of static features at the beginning of each input.
            embedding_size: Integer with the size of the final embeddings.
            input_shape: Tuple with input used to build model.
        """
        super(simple_TCN, self).__init__()

        self.TCN = TCN(n_filter, kernel_size=kernel_size, nb_stacks=n_stack,
                       use_layer_norm=use_LN, dilations=dilatations, activation=activation, dropout_rate=do,
                       use_skip_connections=use_skip)

        self.n_static = n_static
        self.l2_norm = l2_norm
        self.static_DO = tf.keras.layers.Dropout(static_do)
        self.DO_layer = tf.keras.layers.Dropout(do)
        self.LN = tf.keras.layers.LayerNormalization(axis=1)
        if embedding_size:
            self.embedding_size = embedding_size
        else:
            self.embedding_size = n_filter
        self.FC_layer = tf.keras.layers.Dense(self.embedding_size)

        if input_shape:
            self.build(input_shape)

    def call(self, x, training=False):
        """Forward pass function of the encoder.

        We split static features, first channel of the input, from the sequence of variables.
        We pass the sequence in the TCN block and merge both features using a Dense layer.
        We finally project the representation to the unit sphere as in https://arxiv.org/abs/1911.05722.
        Args:
            x: Tensor batch input.
            training: Boolean to pass to the model for handling layers with different
                      behaviour at training time.

        Returns:
            y : The batch of embeddings.
        """

        static, v = tf.split(x, [self.n_static, x.shape[-1] - self.n_static], axis=-1)
        static = static[:, -1, :]
        o = self.TCN(v, training=training)
        static_doed = self.static_DO(static, training=training)
        o = tf.concat([static_doed, o], axis=1)
        o = self.DO_layer(o, training=training)
        o = self.FC_layer(o, training=training)
        y = self.LN(o, training=training)
        if self.l2_norm:
            y = tf.math.l2_normalize(y, axis=-1)

        return y
