import importlib
import typing
import numpy as np
import logging

from soundutils.annotations.audio import Audio


def randomness_decorator(func):
    """ Decorator for randomness """
    def wrapper(self, data: Audio, annotation: typing.Any) -> typing.Tuple[Audio, typing.Any]:
        """ Decorator for randomness and type checking
        Args:
            data (Audio): Audio object to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            data (Audio): Adjusted Audio object
            annotation (typing.Any): Adjusted annotation
        """
        if not isinstance(data, Audio):
            self.logger.error(f"data must be an Audio object, not {type(data)}, skipping augmenting")
            return data, annotation

        if np.random.rand() > self._random_chance:
            return data, annotation

        return func(self, data, annotation)
    return wrapper


class Augmenter:
    """
    Basic class that should be inherited by all augmenters
    """
    def __init__(self,
                 random_chance: float = 0.5,
                 log_level: int = logging.INFO,
                 augment_annotation: bool = False
                 ) -> None:
        self._random_chance = random_chance
        self._log_level = log_level
        self._augment_annotation = augment_annotation

        assert 0 <= self._random_chance <= 1.0, "random_chance must be between 0 and 1.0"

    def augment(self, data: Audio):
        """ Augment the data """
        return NotImplementedError

    @randomness_decorator
    def __call__(self, data: Audio, annotation: typing.Any) -> typing.Tuple[Audio, typing.Any]:
        """
        Randomly add noise to the audio

        Args:
            data (Audio): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted
        Returns:
            data (Audio): Adjusted Audio object
            annotation (typing.Any): Adjusted annotation
        """
        data = self.augment(data)

        if self._augment_annotation and isinstance(annotation, np.ndarray):
            annotation = self.augment(annotation)

        return data, annotation


class RandomAudioNoise(Augmenter):
    """
    Randomly add noise to the audio

    Attributes:
        random_chance (float): Float between 0 and 1.0 setting bounds for random probability. Default value is 0.5.
        log_level (int): Log level for the augmentation. Default value is logging.INFO
        augment_annotation (bool): Whether to augment the annotation. Default value is False.
        max_noise_ratio (float): The maximum ratio to be added to the noise. Default value is 0.1.
    """
    def __init__(self,
                 random_chance: float = 0.5,
                 log_level: int = logging.INFO,
                 augment_annotation: bool = False,
                 max_noise_ratio: float = 0.1
                 ) -> None:
        super(RandomAudioNoise, self).__init__(random_chance, log_level, augment_annotation)
        self._max_noise_ratio = max_noise_ratio

    def augment(self, audio: Audio) -> Audio:
        noise = np.random.uniform(-1, 1, len(audio))
        noise_ratio = np.random.uniform(0, self._max_noise_ratio)
        audio.audio = audio + noise_ratio * noise

        return audio


class RandomAudioPitchShift(Augmenter):
    """
    Randomly shift the pitch of the audio

    Attributes:
        random_chance (float): Float between 0 and 1.0 setting bounds for random probability. Default value is 0.5.
        log_level (int): Log level for the augmentation. Default value is logging.INFO
        augment_annotation (bool): Whether to augment the annotation. Default value is False.
        max_n_steps (int): The maximum number of steps to shift audio. Default value is 5.
    """
    def __init__(self,
                 random_chance: float = 0.5,
                 log_level: int = logging.INFO,
                 augment_annotation: bool = False,
                 max_n_steps: int = 5
                 ) -> None:
        super(RandomAudioPitchShift, self).__init__(random_chance, log_level, augment_annotation)
        self._max_n_steps = max_n_steps

        try:
            self.librosa = importlib.import_module('librosa')
            print("Loaded librosa: ", self.librosa.__version__)
        except ImportError:
            raise ImportError("Librosa is required to augment Audio. Please install it with 'pip install librosa'")

    def augment(self, audio: Audio) -> Audio:
        random_n_steps = np.random.randint(-self._max_n_steps, self._max_n_steps)
        shift_audio = self.librosa.effects.pitch_shift(
            audio.numpy(), sr=audio.sampling_rate, n_steps=random_n_steps, res_type="linear"
        )
        audio.audio = shift_audio

        return audio


class RandomAudioTimeStretch(Augmenter):
    """
    Randomly stretch the audio

    Attributes:
        random_chance (float): Float between 0 and 1.0 setting bounds for random probability. Default value is 0.5.
        log_level (int): Log level for the augmentation. Default value is logging.INFO
        augment_annotation (bool): Whether to augment the annotation. Default value is False.
        min_rate (float): Minimum rate to stretch audio. Default value is 0.8.
        max_rate (float): Maximum rate to stretch audio. Default value is 1.2.
    """
    def __init__(self,
                 random_chance: float = 0.5,
                 log_level: int = logging.INFO,
                 augment_annotation: bool = False,
                 min_rate: float = 0.8,
                 max_rate: float = 1.2
                 ) -> None:
        super(RandomAudioTimeStretch, self).__init__(random_chance, log_level, augment_annotation)
        self._min_rate = min_rate
        self._max_rate = max_rate

        try:
            librosa.__version__
        except ImportError:
            raise ImportError("Librosa is required to augment Audio. Please install it with 'pip install librosa'")

    def augment(self, audio: Audio) -> Audio:
        random_rate = np.random.uniform(self._min_rate, self._max_rate)
        stretch_audio = librosa.effects.time_stretch(audio.numpy(), rate=random_rate)
        audio.audio = stretch_audio

        return audio
