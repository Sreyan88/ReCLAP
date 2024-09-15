import os
import shutil
from pydub.exceptions import CouldntDecodeError
from pydub import AudioSegment
from tqdm import tqdm
from moviepy.editor import VideoFileClip

DATA_DIR = "/mnt/localssd/audio-editing-data/"

all_paths = [
    'WavCaps/Zip_files/BBC_Sound_Effects/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/BBC_Sound_Effects_flac/',
    'WavCaps/Zip_files/FreeSound/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/FreeSound_flac/',
    'WavCaps/Zip_files/SoundBible/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/SoundBible_flac/',
    'WavCaps/Zip_files/AudioSet_SL/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac/',
    'VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/',
]
all_paths = map(lambda x: os.path.join(DATA_DIR, x), all_paths)

def convert_mp4_to_wav(mp4_file, wav_file):
    # Load the MP4 file
    video_clip = VideoFileClip(mp4_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(wav_file)
    video_clip.close()
    audio_clip.close()
    os.remove(file_path)
    
for i in all_paths:
    folder_path = i  # Replace with the actual folder path

    print(f"Processing {folder_path}...")
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.flac'):
            file_path = os.path.join(folder_path, file_name)
            new_file_name = os.path.splitext(file_name)[0] + '.wav'
            new_file_path = os.path.join(folder_path, new_file_name)
            try:
                audio = AudioSegment.from_file(file_path, format="flac")
            except CouldntDecodeError:
                print(f"Error decoding {file_path}. Skipping.")
                continue
            # Export the audio to WAV format
            audio.export(new_file_path, format="wav")
            os.remove(file_path)
        elif file_name.endswith('.mp4'):
            file_path = os.path.join(folder_path, file_name)
            new_file_name = os.path.splitext(file_name)[0] + '.wav'
            new_file_path = os.path.join(folder_path, new_file_name)
            convert_mp4_to_wav(file_path, new_file_path)
