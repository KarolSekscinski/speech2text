import os
import tensorflow as tf
from keras.src.callbacks import Callback

import logging


class Model2onnx(Callback):
    """
    Converts the model to onnx format after training is finished
    """
    def __init__(self,
                 saved_model_path: str,
                 metadata: dict = None,
                 save_on_epoch_end: bool = False
                 ) -> None:
        """
        Converts the model to onnx format after training is finished.
        Args:
            saved_model_path (str): Path to the saved .h5 model.
            metadata (dict, optional): Dictionary containing metadata to be added to the onnx model. Defaults to None.
            save_on_epoch_end (bool, optional): Save the onnx model on every epoch end. Defaults to False.
        """
        super().__init__()
        self._saved_model_path = saved_model_path
        self._metadata = metadata
        self._save_on_epoch_end = save_on_epoch_end

        try:
            import tf2onnx
        except Exception:
            raise ImportError("tf2onnx is not installed. Please install it using 'pip install tf2onnx'")

        try:
            import onnx
        except Exception:
            raise ImportError("onnx is not installed. Please install it using 'pip install onnx'")

    @staticmethod
    def model2onnx(model: tf.keras, onnx_model_path: str) -> None:
        try:
            import tf2onnx

            # convert the model to onnx format
            tf2onnx.convert.from_keras(model, output_path=onnx_model_path)
        except Exception as e:
            print(e)

    @staticmethod
    def include_metadata(onnx_model_path: str, metadata: dict = None) -> None:
        try:
            if metadata and isinstance(metadata, dict):
                import onnx
                # Load the onnx model
                onnx_model = onnx.load(onnx_model_path)

                # Add the metadata dictionary to the model's metadata_props attribute
                for key, value in metadata.items():
                    meta = onnx_model.metadata_props.add()
                    meta.key = key
                    meta.value = str(value)

                # Save the modified onnx model
                onnx.save(onnx_model, onnx_model_path)
        except Exception as e:
            print(e)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Converts the model to onnx format on every epoch end.
        """
        if self._save_on_epoch_end:
            self.on_train_end(logs=logs)

    def on_train_end(self, logs=None):
        """
        Converts the model to onnx format on every epoch end.
        """
        self.model.load_weights(self._saved_model_path)
        onnx_model_path = self._saved_model_path.replace(".h5", ".onnx")
        self.model2onnx(self.model, onnx_model_path)
        self.include_metadata(onnx_model_path, self._metadata)


class TrainLogger(Callback):
    """
    Logs training metrics to a file
    Args:
        log_path (str): Path to the directory where the log file will be saved.
        log_file (str, optional): Name of the log file. Defaults to 'logs.log'.
        log_level (int, optional): Logging level. Defaults to logging.INFO.
    """
    def __init__(self, log_path: str, log_file: str = "logs.log", log_level=logging.INFO, console_output=False) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_file = log_file

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        self.file_handler = logging.FileHandler(os.path.join(self.log_path, self.log_file))
        self.file_handler.setLevel(log_level)
        self.file_handler.setFormatter(self.formatter)

        if not console_output:
            self.logger.handlers[:] = []

        self.logger.addHandler(self.file_handler)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        epoch_message = f"Epoch {epoch}; "
        logs_message = "; ".join([f"{key}: {value}" for key, value in logs.items()])
        self.logger.info(epoch_message + logs_message)
