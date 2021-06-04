import gin
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LayerNormalization, Layer, BatchNormalization, SpatialDropout1D, Activation


class Decode_Block(Layer):
    """Defines a deconvolution block for the TCN decoder.

    This code was directly adapted from keras-tcn library https://github.com/philipperemy/keras-tcn.
    """

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 **kwargs):
        """Init function

        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(Decode_Block, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'deconv1D_{}'.format(k)
                with K.name_scope(name):
                    self._add_and_activate_layer(tf.keras.layers.Conv1DTranspose(filters=self.nb_filters,
                                                                                 kernel_size=self.kernel_size,
                                                                                 dilation_rate=self.dilation_rate,
                                                                                 padding=self.padding,
                                                                                 name=name,
                                                                                 kernel_initializer=self.kernel_initializer))

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._add_and_activate_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._add_and_activate_layer(LayerNormalization())

                self._add_and_activate_layer(Activation('relu'))
                self._add_and_activate_layer(SpatialDropout1D(rate=self.dropout_rate))

            for layer in self.layers:
                self.__setattr__(layer.name, layer)

            super(Decode_Block, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Returns: The output of the deconvolution block.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            x = layer(x, training=training)
            self.layers_outputs.append(x)
        return x

    def compute_output_shape(self, input_shape):
        return self.res_output_shape


@gin.configurable("mirror_TCN_decoder")
class mirror_TCN_decoder(tf.keras.Model):
    """
    Sequence decoder to mirror the TCN architectures from https://arxiv.org/pdf/1803.01271.pdf.
    """

    def __init__(self, kernel_size=gin.REQUIRED, n_filter=gin.REQUIRED, nb_modalities=gin.REQUIRED,
                 dilatations=(1, 2, 4, 8, 16, 32), do=0.0, activation='relu', use_LN=True, input_shape=None,
                 n_static=gin.REQUIRED, seq_length=gin.REQUIRED):
        """Init function.

        Args:
            kernel_size: Size for the time axis of the kernel.
            n_filter: Integer with the number of inner filters used.
            nb_modalities: Integer with the final channel dimension.
            dilatations: List of successive dilatations rates.
            do: Float representing dropout within TCN.
            activation: String with the activation name.
            use_LN: Boolean to use or not LayerNorm.
            input_shape: Tuple with the input shape to network.
            n_static: Integer with the number of static to reconstruct.
            seq_length: Integer with size of sequence which is <= to the receptive field.
        """
        super(mirror_TCN_decoder, self).__init__()
        self.decode_blocks = []
        self.n_static = n_static
        for i, d in enumerate(dilatations[::-1]):
            if i < len(dilatations) - 1:
                self.decode_blocks.append(Decode_Block(dilation_rate=d,
                                                       nb_filters=n_filter,
                                                       kernel_size=kernel_size,
                                                       padding='valid',
                                                       activation=activation,
                                                       dropout_rate=do,
                                                       use_layer_norm=use_LN))
            else:
                self.decode_blocks.append(Decode_Block(dilation_rate=d,
                                                       nb_filters=nb_modalities - n_static,
                                                       kernel_size=kernel_size,
                                                       padding='valid',
                                                       activation=activation,
                                                       dropout_rate=do,
                                                       use_layer_norm=use_LN))
        self.fs_layer = tf.keras.layers.Dense(n_filter + n_static)
        self.n_filter = n_filter
        self.seq_length = seq_length
        if input_shape:
            self.build(input_shape)

    def call(self, x, training=False):
        """Forward pass function of the decoder.

        We aim to reconstruct two elements, sequences of variables and static input.
       The architecture is a mirrored, thus first a dense layer from which we reconstruct static.
       Then a deconvolution block to reconstruct sequence.

        Args:
            x: Tensor batch input.
            training: Boolean to pass to the model for handling layers with different
                      behaviour at training time.

        Returns:
            static: The static feature reconstruction.
            sequence: The sequence reconstruction.
        """
        f = self.fs_layer(x, training=training)
        static, o = tf.split(f, (self.n_static, self.n_filter), axis=-1)
        for decode_block in self.decode_blocks:
            o = decode_block(o, training=training)
        return static[:, 0, :], o[:, :self.seq_length, :]
