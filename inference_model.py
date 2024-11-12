import typing
import numpy as np

from soundutils.inference_model import OnnxInferenceModel
from soundutils.preprocess import WavReader
from soundutils.text_utils import ctc_decoder, get_cer, get_wer


class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._char_list = char_list

    def predict(self, data: np.ndarray):
        data_pred = np.expand_dims(data, axis=0)

        predictions = self.model.run(self.output_names, {self.input_names[0]: data_pred})[0]

        predicted_text = ctc_decoder(predictions, self._char_list)[0]

        return predicted_text


if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    from soundutils.configs import BaseModelConfig
    # TODO
    configs = BaseModelConfig.load("PATH_TO_YAML")

    model = WavToTextModel(model_path=configs.model_path,
                           char_list=configs.vocab,
                           force_cpu=False)
    # TODO
    df = pd.read_csv("PATH_TO_DATA").values.tolist()

    accum_cer, accum_wer = [], []
    for wav_path, label in tqdm(df):
        wav_path = wav_path.replace("\\", "/")
        spectrogram = WavReader.get_spectrogram(wav_path, frame_length=configs.frame_length,
                                                frame_step=configs.frame_step, fft_length=configs.fft_length)
        WavReader.plot_raw_audio(wav_path, label)

        padded_spectrogram = np.pad(spectrogram, ((0, configs.max_spectrogram_length - spectrogram.shape[0]), (0, 0)),
                                    mode="constant", constant_values=0)

        WavReader.plot_spectrogram(spectrogram, label)

        text = model.predict(padded_spectrogram)

        true_label = "".join([l for l in label.lower() if l in configs.vocab])

        cer = get_cer(text, true_label)
        wer = get_wer(text, true_label)

        accum_cer.append(cer)
        accum_wer.append(wer)

    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")


