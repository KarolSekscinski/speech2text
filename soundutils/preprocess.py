import importlib
import io
import typing

import fsspec
from gcsfs.core import GCSFile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def import_librosa(object) -> None:
    """Import librosa using importlib"""
    try:
        version = object.librosa.__version__
    except:
        version = "Librosa version not found"
        try:
            object.librosa = importlib.import_module('librosa')
            print("Imported librosa", object.librosa.__version__)
        except:
            raise ImportError("Librosa is required to augment Audio. Please install it with 'pip install librosa'")


class WavReader:
    """
    Read wav file with librosa and return audio and label

    Attributes:
        frame_length (int) : length of the frames in samples
        frame_step (int) : step size between frames in samples
        fft_length (int) : length of the FFT components
    """
    def __init__(self,
                 frame_length: int = 256,
                 frame_step: int = 160,
                 fft_length: int = 384,
                 *args, **kwargs
                 ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        matplotlib.interactive(False)
        import_librosa(self)

    @staticmethod
    def load_audio(path_or_file, sr: int = None):
        # Check if input is a GCS file handle or a path
        if isinstance(path_or_file, GCSFile):
            # Read from GCS file object
            with io.BytesIO(path_or_file.read()) as audio_binary:
                audio, sample_rate = WavReader.librosa.load(audio_binary, sr=sr)
        elif isinstance(path_or_file, str) and path_or_file.startswith("gs://"):
            fs = fsspec.filesystem("gcs")
            with fs.open(path_or_file, "rb") as f:
                with io.BytesIO(f.read()) as audio_binary:
                    audio, sample_rate = WavReader.librosa.load(audio_binary, sr=sr)
        elif isinstance(path_or_file, str):
            # Local path handling
            audio, sample_rate = WavReader.librosa.load(path_or_file, sr=sr)
        else:
            raise ValueError(f"Unsupported input type: {type(path_or_file)}")

        return audio, sample_rate

    @staticmethod
    def get_spectrogram(wav_path: str, frame_length: int, frame_step: int, fft_length: int) -> np.ndarray:
        """Compute the spectrogram of a wav file"""
        import_librosa(WavReader)

        # Load the wav file and store the audio data in the variable 'audio'
        audio, original_sample_rate = WavReader.load_audio(wav_path, 32000)

        # Compute the Short Time Fourier Transform (STFT) of the audio data
        # STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples and 'fft_length' FFT components
        # The resulting spectrogram is also transposed for convenience
        spectrogram = WavReader.librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T

        # Take the abs of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation,
        # a small value of 1e-10 is added to the denominator to prevent division by zero.
        division_by_zero_prevention = 1e-10
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + division_by_zero_prevention)

        return spectrogram

    @staticmethod
    def plot_raw_audio(wav_path: str, title: str = None, sr: int = 16000) -> None:
        """Plot the raw audio of a wav file"""
        import_librosa(WavReader)

        # Load the wav file with sample rate in 'sr'
        audio, original_sample_rate = WavReader.librosa.load(wav_path, sr=sr)

        duration = len(audio) / original_sample_rate

        time = np.linspace(0, duration, num=len(audio))

        plt.figure(figsize=(15, 5))
        plt.plot(time, audio)
        plt.title(title) if title else plt.title("Audio Plot")
        plt.xlabel("time (s)")
        plt.ylabel("Signal wave")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrogram(spectrogram: np.ndarray, title: str = "", transpose: bool = True, invert: bool = True) -> None:
        """Plot the spectrogram of a wav file"""
        if transpose:
            spectrogram = np.transpose(spectrogram)

        if invert:
            spectrogram = spectrogram[::-1]

        plt.figure(figsize=(15, 5))
        plt.imshow(spectrogram, aspect="auto", origin="lower")
        plt.title(f"Spectrogram: {title}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def __call__(self, audio_path: str, label: typing.Any):
        """Extract the spectrogram and label of a wav file"""
        return self.get_spectrogram(audio_path, self.frame_length, self.frame_step, self.fft_length), label
