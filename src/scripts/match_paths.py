"""Match Paths

Matches the paths of the wav files for the dataset.csv

Created by onieto@adobe.com, 1/4/24
"""

import glob
import os
from tqdm import tqdm
import pandas as pd

AUDIOEDIT_DATA_PATH = "/home/onieto/audio-editing-data"
FINAL_AUDIOEDIT_DATA_PATH = "/mnt/localssd/audio-editing-data"
OUTPUT_CSV = "onieto_dataset.csv"


def get_audioset_wav_paths(data_dir):
    print("Reading AudioSet paths...")
    balanced_files = audioset_files = glob.glob(
            os.path.join(data_dir, "AudioSet-v1-Wavs", "data", "balanced_train_segments", "audio", "*.wav"))
    unbalanced_files = glob.glob(
            os.path.join(data_dir, "AudioSet-v1-Wavs", "data", "unbalanced_train_segments", "audio", "*.wav"))
    eval_files = glob.glob(
            os.path.join(data_dir, "AudioSet-v1-Wavs", "data", "eval_segments", "audio", "*.wav"))
    return balanced_files + unbalanced_files + eval_files


def process():

    print("Reading main csv...")
    data_df = pd.read_csv(os.path.join(AUDIOEDIT_DATA_PATH, "dataset.csv"))

    # Prepare AudioSet
    audioset_wav_paths = get_audioset_wav_paths(AUDIOEDIT_DATA_PATH)
    audioset_wav_paths_dict = {os.path.basename(x)[:11]: x for x in audioset_wav_paths}

    print("Matching all datasets...")
    updated_rows = []
    k = 0
    for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        ds = row["dataset"]
        if ds == "audioset" or ds == "audiocaps" or ds == "musiccaps":
            try:
                path = audioset_wav_paths_dict[row["path"].replace(".wav", "")]
            except KeyError:
                path = row["path"]
        elif ds == "medley_solo":
            path = row["path"].replace("medley_solo", "medley_solo/resampled")
            path = os.path.join(AUDIOEDIT_DATA_PATH, path)
        elif ds == "AudioSet_SL":
            path = os.path.join(
                AUDIOEDIT_DATA_PATH,
                "WavCaps/Zip_files/AudioSet_SL/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac",
                os.path.basename(row["path"]))
        elif ds == "vgg":
            path = row["path"].replace("VGGSound/VGGSound/wavs", "VGGSound") + ".wav"
            path = os.path.join(AUDIOEDIT_DATA_PATH, path)
        elif ds == "BBC" or \
                ds == "Clotho" or \
                ds == "MACS" or \
                ds == "SoundBible" or \
                ds == "esc50" or \
                ds == "freesound" or \
                ds == "fsd50k" or \
                ds == "gtzan" or \
                ds == "musical-instrument" or \
                ds == "soniss" or \
                ds == "urban-sound" or \
                ds == "wavtext5k":
            path = os.path.join(AUDIOEDIT_DATA_PATH, row["path"])
        else:
            raise RuntimeError(f"Not supported dataset id ({ds})")

        if not os.path.isfile(path):
            k += 1
            print(f"{path} not found")
            continue

        path = path.replace(
            AUDIOEDIT_DATA_PATH, 
            FINAL_AUDIOEDIT_DATA_PATH)
        row["path"] = path
        updated_rows.append(row)


    print(f"{k} files not found")
    out_df = pd.DataFrame(updated_rows)
    out_df.to_csv(OUTPUT_CSV, header=list(data_df.columns), index=None)


if __name__ == "__main__":
    process()

