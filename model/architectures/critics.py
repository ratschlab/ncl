import gin
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, LayerNormalization
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx('float32')


@gin.configurable("non_linear_projector")
def non_linear_projector(embedding_size, projection_size):
    """
    Projector before critic used in https://arxiv.org/pdf/2002.05709.pdf.
    Args:
        embedding_size: Integer representing representation size
        projection_size: Integer representing projection size

    Returns:
        tf.keras.Model instance
    """
    x = Input((embedding_size,))
    o = Dense(projection_size, activation='linear')(x)
    o = LayerNormalization()(o)
    o = Activation('relu')(o)
    y = Dense(projection_size, activation='linear')(o)
    return Model(inputs=x, outputs=y)
