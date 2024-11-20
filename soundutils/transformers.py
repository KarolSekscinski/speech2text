import logging
import typing
import numpy as np
import importlib
import threading

from soundutils.annotations.audio import Audio


class Transformer:
    def __init__(self, log_level: int = logging) -> None:
        self._log_level = log_level

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(self._log_level)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs) -> typing.Any:
        raise NotImplementedError


class ExpandDimsTransformer(Transformer):
    def __init__(self, axis: int = -1) -> None:
        self._axis = axis

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.expand_dims(data, axis=self._axis), label


class LabelIndexerTransformer(Transformer):
    def __init__(self,
                 vocabulary: typing.List[str],
                 ) -> None:
        self._vocabulary = vocabulary

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self._vocabulary.index(l) for l in label if l in self._vocabulary])


class LabelPaddingTransformer(Transformer):
    def __init__(self,
                 padding_value: int,
                 max_word_length: int = None,
                 use_of_batch: bool = False
                 ) -> None:
        self._padding_value = padding_value
        self._max_word_length = max_word_length
        self._use_of_batch = use_of_batch

        if not use_of_batch and max_word_length is None:
            raise ValueError("max_word_length must be specified if use_of_batch is False")

    def __call__(self, data: np.ndarray, label: np.ndarray):
        if self._use_of_batch:
            max_len = max([len(a) for a in label])
            padded_labels = []
            for l in label:
                padded_label = np.pad(l, (0, max_len - len(l)), "constant", constant_values=self._padding_value)
                padded_labels.append(padded_label)

            padded_labels = np.array(padded_labels)
            return data, padded_labels

        label = label[:self._max_word_length]
        return data, np.pad(label,
                            (0, self._max_word_length - len(label)),
                            "constant",
                            constant_values=self._padding_value)


class SpectrogramPaddingTransformer(Transformer):
    def __init__(self,
                 padding_value: int,
                 max_spectrogram_length: int = None,
                 use_of_batch: bool = False
                 ) -> None:
        self._padding_value = padding_value
        self._max_spectrogram_length = max_spectrogram_length
        self._use_of_batch = use_of_batch

        if not use_of_batch and max_spectrogram_length is None:
            raise ValueError("max_spectrogram_length must be specified if use_of_batch is False")

    def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
        if self._use_of_batch:
            max_len = max([len(a) for a in spectrogram])
            padded_spectrograms = []
            for s in spectrogram:
                padded_spectrogram = np.pad(s,
                                            ((0, max_len - s.shape[0]),
                                             (0, 0)),
                                            "constant",
                                            constant_values=self._padding_value)
                padded_spectrograms.append(padded_spectrogram)

            padded_spectrograms = np.array(padded_spectrograms)
            label = np.array(label)

        padded_spectrogram = np.pad(spectrogram,
                                    ((0, self._max_spectrogram_length - spectrogram.shape[0]),
                                     (0, 0)),
                                    "constant",
                                    constant_values=self._padding_value)
        return padded_spectrogram, label


class AudioPaddingTransformer(Transformer):
    def __init__(self,
                 max_audio_length: int,
                 padding_value: int = 0,
                 use_of_batch: bool = False,
                 limit: bool = False):
        super(AudioPaddingTransformer, self).__init__()
        self._max_audio_length = max_audio_length
        self._padding_value = padding_value
        self._use_of_batch = use_of_batch
        self._limit = limit

    def __call__(self, audio: Audio, label: typing.Any):
        if self._use_of_batch:
            max_len = max([len(a) for a in audio])
            padded_audios = []
            for a in audio:
                padded_audio = np.pad(a,
                                      (0, max_len - a.shape[0]),
                                      "constant",
                                      constant_values=self._padding_value)
                padded_audios.append(padded_audio)

            padded_audio = np.array(padded_audios)
            if self._limit:
                padded_audios = padded_audios[:, :self._max_audio_length]
            return padded_audios, label

        audio_numpy = audio.numpy()
        if self._limit:
            audio_numpy = audio_numpy[:self._max_audio_length]
        padded_audio = np.pad(audio_numpy,
                              (0, self._max_audio_length - audio_numpy.shape[0]),
                              "constant",
                              constant_values=self._padding_value)
        audio.audio = padded_audio
        return audio, label


class AudioToSpectrogramTransformer(Transformer):
    def __init__(self,
                 frame_length: int = 256,
                 frame_step: int = 160,
                 fft_length: int = 384,
                 log_level: int = logging.INFO
                 ) -> None:
        super(AudioToSpectrogramTransformer, self).__init__(log_level=log_level)
        self._frame_length = frame_length
        self._frame_step = frame_step
        self._fft_length = fft_length

        try:
            self.librosa = importlib.import_module('librosa')
            print("librosa version: ", self.librosa.__version__)
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with: 'pip install librosa'")

    def __call__(self, audio: Audio, label: typing.Any):
        spectrogram = self.librosa.stft(audio.numpy(),
                                        hop_length=self._frame_step,
                                        win_length=self._frame_length,
                                        n_fft=self._fft_length).T
        spectrogram = np.abs(spectrogram)

        spectrogram = np.power(spectrogram, 0.5)

        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)
        return spectrogram, label
