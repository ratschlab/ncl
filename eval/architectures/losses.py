import tensorflow as tf
import tensorflow_addons as tfa
import gin


gin.external_configurable(tf.keras.losses.SparseCategoricalCrossentropy, name='cross_entropy')
gin.external_configurable(tf.keras.losses.MeanSquaredError, name='mse')

