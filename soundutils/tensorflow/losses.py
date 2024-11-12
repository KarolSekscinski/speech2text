import tensorflow as tf


class CTCLoss(tf.keras.losses.Loss):
    """ Custom CTC for training the model """
    def __init__(self, name: str = "CTC_loss") -> None:
        super(CTCLoss, self).__init__()
        self._name = name
        self._loss_function = tf.keras.backend.ctc_batch_cost

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        """ Compute the training batch CTC loss value """
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_len = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_len = input_len * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_len = label_len * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self._loss_function(y_true, y_pred, input_len, label_len)
        return loss
