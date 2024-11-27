import tensorflow as tf
import pandas as pd
from tqdm import tqdm

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from soundutils.preprocess import WavReader

from soundutils.tensorflow.data_provider import AudioDataProvider
from soundutils.transformers import LabelIndexerTransformer, LabelPaddingTransformer, SpectrogramPaddingTransformer
from soundutils.tensorflow.losses import CTCLoss
from soundutils.tensorflow.callbacks import Model2onnx, TrainLogger
from soundutils.tensorflow.metrics import CERMetric, WERMetric
from cloud_utils import upload_to_gcs
from configs import ModelConfigs
from model import train_model
import gcsfs


bucket_name = "modelasr-studia1"
dataset_path = f'gs://{bucket_name}/data'
metadata_path = f'gs://{bucket_name}/metadata/metadata.tsv'

fs = gcsfs.GCSFileSystem()

# Read metadata files and parse it
columns = ['path', 'sentence']
with fs.open(metadata_path, 'r') as f:
    metadata_df = pd.read_csv(metadata_path, sep="\t")
metadata_df = metadata_df[columns]

# for dev
metadata_df = metadata_df.head(1000)

# structure the dataset where each row is a list of [wav_file_path, sound transcription]
dataset = [[f"{dataset_path}/{file.replace('.mp3', '.wav')}", label.lower()] for file, label in
           metadata_df.values.tolist()]

# create a ModelConfigs object to store model conf
configs = ModelConfigs()

max_text_length, max_spectrogram_length = 0, 0
for file_path, label in tqdm(dataset):
    with fs.open(file_path, 'rb') as f:
        spectrogram = WavReader.get_spectrogram(f,
                                                frame_length=configs.frame_length,
                                                frame_step=configs.frame_step,
                                                fft_length=configs.fft_length
                                                )
    valid_label = [c for c in label if c in configs.vocab]
    max_text_length = max(max_text_length, len(valid_label))
    max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])
    configs.input_shape = [max_spectrogram_length, spectrogram.shape[1]]

configs.max_spectrogram_length = max_spectrogram_length
configs.max_text_length = max_text_length
configs.save()

# Create a data provider for the dataset
data_provider = AudioDataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(frame_length=configs.frame_length,
                  frame_step=configs.frame_step,
                  fft_length=configs.fft_length
                  )
    ],
    transformers=[
        SpectrogramPaddingTransformer(max_spectrogram_length=configs.max_spectrogram_length,
                                      padding_value=0),
        LabelIndexerTransformer(configs.vocab),
        LabelPaddingTransformer(max_word_length=configs.max_text_length,
                                padding_value=len(configs.vocab))
    ]
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Creating TensorFlow model architecture
model = train_model(
    input_dim=configs.input_shape,
    output_dim=len(configs.vocab),
    dropout=0.5
)

# Compile the model and print the summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCLoss(),
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
    ],
    run_eagerly=False
)

model.summary(line_length=120)

# Define callbacks
early_stopping = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5",
                             monitor="val_CER", verbose=1,
                             save_best_only=True, mode="min")
train_logger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduce_LROnPlateau = ReduceLROnPlateau(monitor="val_CER", factor=0.8,
                                       min_delta=1e-10, patience=5,
                                       verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.training_epochs,
    callbacks=[early_stopping, checkpoint, train_logger, reduce_LROnPlateau, tb_callback, model2onnx],
    workers=configs.train_workers,
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(f"{configs.model_path}/train.tsv")
val_data_provider.to_csv(f"{configs.model_path}/val.tsv")
upload_to_gcs(bucket_name, configs.model_path, configs.model_path)

