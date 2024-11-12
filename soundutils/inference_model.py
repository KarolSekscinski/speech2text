import typing
import os
import time
import numpy as np
import onnxruntime as ort
from collections import deque


class FpsWrapper:
    """
    Decorator class to calculate frames per second of a function
    """
    def __init__(self, func: typing.Callable):
        self._func = func
        self._fps_list = deque([], maxlen=100)

    def __call__(self, *args, **kwargs):
        start = time.time()
        results = self._func(self.instance, *args, **kwargs)
        self._fps_list.append(1 / (time.time() - start))
        self.instance.fps = np.mean(self._fps_list)
        return results

    def __get__(self, instance, owner):
        self.instance = instance
        return self.__call__.__get__(instance, owner)


class OnnxInferenceModel:
    """
    Base class for all inference models that use onnxruntime

    Attributes:
        _model_path (str, optional): Path to the model folder. Defaults to "".
        _force_cpu (bool, optional): Force the model to run on CPU or GPU. Defaults to GPU.
        _default_model_name (str, optional): Default model name. Defaults to "model.onnx".
    """
    def __init__(self,
                 model_path: str = "",
                 force_cpu: bool = False,
                 default_model_name: str = "model.onnx",
                 *args, **kwargs):
        self._model_path = model_path.replace("\\", "/")
        self._force_cpu = force_cpu
        self._default_model_name = default_model_name

        # Check if model path is a directory with os path
        if os.path.isdir(self._model_path):
            self._model_path = os.path.join(self._model_path, self._default_model_name)

        if not os.path.exists(self._model_path):
            raise Exception(f"Model path ({self._model_path}) does not exist")

        providers = ["CUDAExecutionProvider",
                     "CPUExecutionProvider"] if ort.get_device() == "GPU" and not force_cpu else [
            "CPUExecutionProvider"]

        self.model = ort.InferenceSession(self._model_path, providers=providers)

        self._metadata = {}
        if self.model.get_modelmeta().custiom_metadata_map:
            # Add metadata to self object
            for key, value in self.model.get_modelmeta().custom_metadata_map.items():
                try:
                    new_value = eval(value)
                except Exception:
                    new_value = value
                self._metadata[key] = new_value

        # Update providers priority to only CPUExecutionProvider
        if self._force_cpu:
            self.model.set_providers(["CPUExecutionProvider"])

        self.input_shapes = [meta.shape for meta in self.model.get_inputs()]
        self.input_names = [meta.name for meta in self.model._inputs_meta]
        self.output_names = [meta.name for meta in self.model._outputs_meta]

    def predict(self, data: np.ndarray, *args, **kwargs):
        raise NotImplementedError("Not implemented yet")

    @FpsWrapper
    def __call__(self, data: np.ndarray):
        results = self.predict(data)
        return results
