import numpy as np
import pandas as pd
from tqdm import tqdm
import typing
import os
import copy
import logging

from .augmenters import Augmenter
from .transformers import Transformer


class BasicDataProvider:
    def __init__(self,
                 dataset: typing.Union[str, list, pd.DataFrame],
                 data_preprocessors: typing.List[typing.Callable] = None,
                 batch_size: int = 4,
                 shuffle: bool = True,
                 initial_epoch: int = 1,
                 augmenters: typing.List[Augmenter] = None,
                 transformers: typing.List[Transformer] = None,
                 batch_postprocessors: typing.List[typing.Callable] = None,
                 skip_validation: bool = True,
                 limit: int = None,
                 use_cache: bool = False,
                 log_level: int = logging.INFO,
                 numpy: bool = True
                 ) -> None:
        """
        Standardised object for providing data to tensorflow model while training
        Attributes:
            dataset (str, list, pd.DataFrame): Path to the dataset, list of data or pandas dataframe
            data_preprocessors (list): List of data preprocessors. (e.g. [read audio])
            batch_size (int): The number of samples to include in the batch. (default: 4)
            shuffle (bool, optional): Whether to shuffle the dataset. (default: True)
            initial_epoch (int): The initial epoch number. (default: 1)
            augmenters (list, optional): List of augmenters. (default: None)
            transformers (list, optional): List of transformers. (default: None)
            batch_postprocessors (list, optional): List of batch postprocessors. (default: None)
            skip_validation (bool, optional): Whether to skip the validation. (default: True)
            limit (int, optional): Limit the number of samples in the dataset. (default: None)
            use_cache (bool, optional): Whether to cache the dataset. (default: False)
            log_level (int, optional): Log level. (default: logging.INFO)
            numpy (bool): Whether to use numpy to convert data to numpy. (default: True)
        """
        self._dataset = dataset
        self._data_preprocessors = [] if data_preprocessors is None else data_preprocessors
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._epoch = initial_epoch
        self._augmenters = [] if augmenters is None else augmenters
        self._transformers = [] if transformers is None else transformers
        self._batch_postprocessors = [] if batch_postprocessors is None else batch_postprocessors
        self._skip_validation = skip_validation
        self._limit = limit
        self._use_cache = use_cache
        self._step = 0
        self._cache = {}
        self._on_epoch_end_remove = []
        self._executor = None
        self._numpy = numpy
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Validate dataset
        if not skip_validation:
            self._dataset = self.validate(dataset)
        else:
            self.logger.info("Skipping Dataset validation...")

        # Check if dataset is iterable
        if not len(dataset):
            raise ValueError("Dataset must be iterable")

        if limit:
            self.logger.info(f"Limiting dataset to {limit} samples")
            self._dataset = dataset[:limit]

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(len(self._dataset) / self._batch_size))

    @property
    def augmenters(self) -> typing.List[Augmenter]:
        """ Return augmenters """
        return self._augmenters

    @augmenters.setter
    def augmenters(self, augmenters: typing.List[Augmenter]):
        """ Decorator for adding augmenters to the BasicDataProvider """
        for augmenter in augmenters:
            if isinstance(augmenter, Augmenter):
                if self._augmenters is not None:
                    self._augmenters.append(augmenter)
                else:
                    self._augmenters = [augmenter]
            else:
                self.logger.warning(f"Augmenter {augmenter} is not an instance of Augmenter")

    @property
    def transformers(self) -> typing.List[Transformer]:
        """ Return transformers """
        return self._transformers

    @transformers.setter
    def transformers(self, transformers: typing.List[Transformer]):
        """ Decorator for adding transformers to the BasicDataProvider """
        for transformer in transformers:
            if isinstance(transformer, Transformer):
                if self._transformers is not None:
                    self._transformers.append(transformer)
                else:
                    self._transformers = [transformer]
            else:
                self.logger.warning(f"Transformer {transformer} is not an instance of Transformer")

    @property
    def epochs(self) -> int:
        """ Return current epoch """
        return self._epoch

    @property
    def step(self) -> int:
        """ Return current step """
        return self._step

    def on_epoch_end(self):
        """ Shuffle training dataset and increment epoch counter at the end of each epoch. """
        self._epoch += 1
        if self._shuffle:
            np.random.shuffle(self._dataset)

        # Remove any samples that were marked for removal
        for remove in self._on_epoch_end_remove:
            self.logger.warning(f"Removing {remove} from dataset.")
            self._dataset.remove(remove)
        self._on_epoch_end_remove = []

    def validate_list_dataset(self, dataset: list) -> list:
        """ Validate dataset """
        validated_data = [data for data in tqdm(dataset, desc="Validating dataset") if os.path.exists(data[0])]
        if not validated_data:
            raise FileNotFoundError("No valid data found in dataset")
        return validated_data

    def validate(self, dataset: typing.Union[str, list, pd.DataFrame]) -> typing.Union[list, str]:
        """ Validate dataset and return the dataset """
        if isinstance(dataset, str):
            if os.path.exists(dataset):
                return dataset
        elif isinstance(dataset, list):
            return self.validate_list_dataset(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return self.validate_list_dataset(dataset.values.tolist())
        else:
            raise ValueError(f"Dataset must be a path, list or pandas dataframe, but it is {type(dataset)}")

    def split(self, split: float = 0.9, shuffle: bool = True) -> typing.Tuple[typing.Any, typing.Any]:
        """
        Split current data provider into training and validation data providers

        Args:
            split (float, optional): Split ratio. (default: 0.9)
            shuffle (bool, optional): Whether to shuffle the dataset. (default: True)

        Returns:
            train_data_provider (tf.keras.utils.Sequence): Training data provider
            validation_data_provider (tf.keras.utils.Sequence): Validation data provider
        """
        if shuffle:
            np.random.shuffle(self._dataset)

        train_data_provider, validation_data_provider = copy.deepcopy(self), copy.deepcopy(self)
        train_data_provider._dataset = self._dataset[:int(len(self._dataset) * split)]
        validation_data_provider._dataset = self._dataset[int(len(self._dataset) * split):]

        return train_data_provider, validation_data_provider

    def save_to_csv(self, path: str, index: bool = False) -> None:
        """
        Save the dataset to a csv file

        Args:
            path (str, optional): The path to save the csv file
            index (bool, optional): Whether to save the index. (default: False)
        """
        df = pd.DataFrame(self._dataset)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=index)

    def get_batch_annotations(self, index: int) -> typing.List:
        """
        Returns a batch of annotations by batch index in the dataset

        Args:
            index (int, optional): The index of the batch in

        Returns:
            batch_annotations (list): A list of batch annotations
        """
        self._step = index
        start_index = index * self._batch_size

        # Get batch indexes
        batch_indexes = [i for i in range(start_index, start_index + self._batch_size) if i < len(self._dataset)]

        # Read batch data
        batch_annotations = [self._dataset[index] for index in batch_indexes]

        return batch_annotations

    def start_executor(self) -> None:
        """ Start the executor to process data. """
        def executor(batch_data):
            for data in batch_data:
                yield self.process_data(data)

        if not hasattr(self, "_executor"):
            self._executor = executor

    def __iter__(self):
        """ Create a generator that iterate over the Sequence """
        for index in range(len(self)):
            results = self[index]
            yield results

    def process_data(self, batch_data):
        """ Process data batch of data """
        if self._use_cache and batch_data[0] in self._cache and isinstance(batch_data[0], str):
            data, annotation = copy.deepcopy(self._cache[batch_data[0]])
        else:
            data, annotation = batch_data
            for preprocessor in self._data_preprocessors:
                data, annotation = preprocessor(data, annotation)

            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, marking for removal on epoch end.")
                self._on_epoch_end_remove.append(batch_data)
                return None, None

            if self._use_cache and batch_data[0] not in self._cache:
                self._cache[batch_data[0]] = (copy.deepcopy(data), copy.deepcopy(annotation))

        # then augment, transform and postprocess the batch data
        for objects in [self.augmenters, self.transformers]:
            for _object in objects:
                data, annotation = _object(data, annotation)

        if self._numpy:
            try:
                data = data.numpy()
                annotation = annotation.numpy()
            except Exception:
                pass
        return data, annotation

    def __getitem__(self, index: int):
        """
        Returns a batch fo processed data by index

        Args:
            index (int): Index of batch

        Returns:
            tuple: batch of data and batch of annotations
        """
        if index == 0:
            self.start_executor()
        dataset_batch = self.get_batch_annotations(index)

        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for data, annotation in self._executor(dataset_batch):
            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, skipping..")
                continue
            batch_data.append(data)
            batch_annotations.append(annotation)

        if self._batch_postprocessors:
            for batch_postprocessor in self._batch_postprocessors:
                batch_data, batch_annotations = batch_postprocessor(batch_data, batch_annotations)

            return batch_data, batch_annotations

        try:
            return np.array(batch_data), np.array(batch_annotations)
        except Exception:
            return batch_data, batch_annotations
