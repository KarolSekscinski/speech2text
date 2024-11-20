import tensorflow as tf
from ..data_provider import BasicDataProvider


class AudioDataProvider(BasicDataProvider, tf.keras.utils.Sequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
