import gin
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, LayerNormalization
from tensorflow.keras.models import Model


@gin.configurable("MLP_classifier")
def MLP_classifier(embedding_shape=gin.REQUIRED, nb_class=10, do=0.0, add_layer=0, inner_size=None, bias_init=None,
                   name='MLP_classifier'):
    """Modular MLP Downstream task classifier.

    Args:
        embedding_shape: Tuple with shape of the embedding.
        nb_class: Integer for the number of class in the downstream task.
        do: (Optional) Float with value of eventual dropout used when more that 2 layer.
        add_layer: (Optional) Integer with the number of supplementary layers. If 0 the classifier is a 2-layer mlp.
        inner_size: (Optional) Integer for the size of the layer. If None, set to embedding_size.
        bias_init: (Optional) List for the bias init if provided.
        name: (Optional) String with the name of the classifer.

    Returns:
        a tf.keras.Model corresponding to the "add on" classifier on top
        of the previously learned representation.
    """

    """Modular MLP Downstream task classifier.

    Args:
        embedding_shape: Tuple with the learned representation shape.
        nb_class: Integer for the number of units in the last layer.
        name: (Optional) Name for the tf.keras.Model outputted.
        do: Float for drop out parameter.
        add_layer: How many layer to add to classic network for depth testing.

    Returns:
        a tf.keras.Model corresponding to the "add on" classifier on top
        of the previously learned representation.
    """
    x = Input(embedding_shape)
    if inner_size:
        f = Dense(inner_size)(x)
    else:
        f = Dense(embedding_shape[0])(x)
    f = LayerNormalization()(f)
    f = Activation('relu')(f)
    if add_layer != 0:
        f = Dropout(do)(f)

    # For testing bigger MLP
    for k in range(add_layer):
        if inner_size:
            f = Dense(inner_size)(f)
        else:
            f = Dense(embedding_shape[0])(f)
        f = LayerNormalization()(f)
        f = Activation('relu')(f)
        if k != add_layer - 1:
            f = Dropout(do)(f)
    if not bias_init:
        o = Dense(nb_class, activation='softmax')(f)
    else:
        output_bias = tf.keras.initializers.Constant(bias_init)
        o = Dense(nb_class, activation='softmax', bias_initializer=output_bias)(f)

    return Model(inputs=x, outputs=o, name=name)


@gin.configurable("Linear_classifier")
def Linear_classifier(embedding_shape=gin.REQUIRED, nb_class=10, bias_init=None, name='Linear_classifier'):
    """Linear classifier used to eval representations.

    Linear classifier is equivalent to a Logistic regression in the case of 2 classes.

    Args:
        embedding_shape: Tuple representing the learned representation shape.
        nb_class: Integer for the number of units in the last layer.
        bias_init: (Optional) List for the bias init if provided.
        name: (Optional) Name for the tf.keras.Model outputted.

    Returns:
        a tf.keras.Model corresponding to the "add on" classifier on top
        of the previously learned representation.
    """
    x = Input(embedding_shape)
    if not bias_init:
        o = Dense(nb_class, activation='softmax')(x)
    else:
        output_bias = tf.keras.initializers.Constant(bias_init)
        o = Dense(nb_class, activation='softmax', bias_initializer=output_bias)(x)
    return Model(inputs=x, outputs=o, name=name)
