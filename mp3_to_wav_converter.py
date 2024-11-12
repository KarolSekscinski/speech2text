import os
from pydub import AudioSegment
from tqdm import tqdm


def mp3_to_wav(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for filename in tqdm(os.listdir(input_path)):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(input_path, filename)
            wav_path = os.path.join(output_path, filename.replace(".mp3", ".wav"))
            try:
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format="wav")
            except Exception as e:
                print(f"Plik {mp3_path} jest zly")
            finally:
                os.remove(mp3_path)


path_to_mp3_dir = "D:/Dokumenty/en~/clips"
path_to_wav_dir = "D:/Dokumenty/en~/clips_wav"
mp3_to_wav(path_to_mp3_dir, path_to_wav_dir)
