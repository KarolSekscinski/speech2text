import tensorflow as tf
from keras import layers


def activation_layer(layer, activation: str="relu", alpha: float=0.1) -> tf.Tensor:
    """ Activation layer wrapper for LeakyReLU and ReLU activation functions
    Args:
        layer: tf.Tensor
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)
    Returns:
        tf.Tensor
    """
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        layer = layers.LeakyReLU(alpha=alpha)(layer)

    return layer

