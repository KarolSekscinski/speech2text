import os
import numpy as np


class Audio:
    """
    Audio object

    Attributes:
        _audio (np.ndarray): Audio array
        sample_rate (int): Sample rate
        init_successful (bool): True if audio was successfully read
        library (object): Library used to read audio file

    """
    init_successful = False
    augmented = False

    def __init__(self,
                 audio_path: str,
                 sample_rate: int = 16000,
                 library=None) -> None:
        if library is None:
            raise ValueError("Library is required (e.g. librosa object)")

        if isinstance(audio_path, str):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file {audio_path} not found")
            self._audio, self._sample_rate = library.load(audio_path, sr=sample_rate)
            self.path = audio_path
            self.init_successful = True
        else:
            raise TypeError(f"audio_path must be a path to an audio file, not {type(audio_path)}")

    @property
    def audio(self) -> np.ndarray:
        return self._audio

    @audio.setter
    def audio(self, value: np.ndarray) -> None:
        self.augmented = True
        self._audio = value

    @property
    def shape(self) -> tuple:
        return self.audio.shape

    def numpy(self) -> np.ndarray:
        return self._audio

    def __add__(self, other: np.ndarray) -> np.ndarray:
        self._audio = self.audio + other
        self.init_successful = True
        return self

    def __len__(self) -> int:
        return len(self._audio)

    def __call__(self) -> np.ndarray:
        return self._audio

    def __repr__(self):
        return repr(self._audio)

    def __array__(self):
        return self._audio
