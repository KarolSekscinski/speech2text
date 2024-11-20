import os
import pandas as pd

# Path to the directory containing WAV files
directory_path = "D:/Dokumenty/en~/clips_wav"

# Path to the metadata CSV
csvs = ["validated.tsv", "test.tsv", "dev.tsv", "other.tsv"]
path_to_metadata = "D:/Dokumenty/en~/"

# save new csv
path_to_save = "D:/Dokumenty/en~/metadata.tsv"

# List all .wav files in directory A
wav_files = {file for file in os.listdir(directory_path) if file.endswith('.wav')}

final_df = {
    "client_id": [],
    "path": [],
    "sentence": [],
    "up_votes": [],
    "down_votes": [],
    "age": [],
    "gender": [],
    "accent": []
}
final_df = pd.DataFrame(final_df)
for csv in csvs:
    metadata = pd.read_csv(os.path.join(path_to_metadata, csv), sep="\t")
    # Filter metadata to keep only rows where file_path exists in wav_files
    metadata['file_name'] = metadata['path'].apply(os.path.basename)  # Extract file names
    metadata['file_name'] = metadata['file_name'].apply(lambda x: x.replace(".mp3", ".wav"))
    filtered_metadata = metadata[metadata['file_name'].isin(wav_files)]
    # Drop the temporary 'file_name' column
    filtered_metadata = filtered_metadata.drop(columns=['file_name'])
    print(metadata.shape)
    print(filtered_metadata.shape)
    final_df = pd.concat([final_df, filtered_metadata], ignore_index=True)

columns = ['path', 'sentence']

df = final_df[columns]
dataset = []
files_to_delete = []
for file, label in df.values.tolist():
    try:
        dataset.append([f"{file.replace('.mp3', '.wav')}", label.lower()])
    except AttributeError as e:
        files_to_delete.append(file)


# Example Condition: Keep rows where 'path' is not in files_to_delete
condition = ~df['path'].isin(files_to_delete)

# Filter the DataFrame based on the condition
df_filtered = df[condition]

# Save the updated metadata.csv
df_filtered.to_csv(path_to_save, index=False, sep="\t")

print(f"Updated metadata saved to {path_to_save}.")

