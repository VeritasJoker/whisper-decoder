import os
import numpy as np
import scipy.io.wavfile as wavfile
from utils import load_transcript, load_label, load_audio


def split_audio_word(data_dir, audioname, filename, saving_dir):
    # load audio
    fs, full_audio = load_audio(os.path.join(data_dir, audioname))

    if "label" in saving_dir:  # label_df
        df = load_label(os.path.join(data_dir, filename))
    elif "transcript" in saving_dir:  # transcript_df
        df = load_transcript(os.path.join(data_dir, filename))
    else:
        return None
    breakpoint()
    for i in df.index:
        if "label" in saving_dir:
            onset = df.onset_sec[i]
            offset = df.offset_sec[i]
        else:
            onset = df.audio_onset[i]
            offset = df.audio_offset[i]

        # Split wav
        chunk_data = full_audio[int(onset * fs) : int(offset * fs)]
        chunk_name = os.path.join(saving_dir, f"segment_{i:04d}-{df.word[i]}.wav")
        wavfile.write(chunk_name, fs, chunk_data)

    return None


def split_audio_word_context(data_dir, audioname, filename, saving_dir):
    # load audio
    fs, full_audio = load_audio(os.path.join(data_dir, audioname))

    if "label" in saving_dir:
        df = load_label(os.path.join(data_dir, filename))
    elif "transcript" in saving_dir:
        df = load_transcript(os.path.join(data_dir, filename))
    else:
        return None

    df["ctx_idx"] = 0
    # sliding window of audio segements
    for i in df.index:
        end_onset = df.offset_sec[i]
        start_onset = np.max([df.iloc[0]["onset_sec"], (end_onset - 30)])

        # Extract audio segment and save
        chunk_data = full_audio[int(start_onset * fs) : int(end_onset * fs)]
        chunk_name = os.path.join(saving_dir, f"segment_{i:04d}-{df.word[i]}.wav")
        wavfile.write(chunk_name, fs, chunk_data)

        # Compute earliest index in the context
        df.loc[i, "ctx_idx"] = df.onset_sec.ge(start_onset).idxmax()

        if end_onset >= 40:
            breakpoint()

    return None


def main():
    # project = "tfs"
    # label_name = ""
    # trans_name = ""
    # audio_name = ""

    project = "podcast"
    label_name = "777_full_labels.pkl"
    trans_name = "podcastAlignedDatum.txt"
    audio_name = "Podcast.wav"

    data_dir = os.path.join("data", project)
    label_split_folder = os.path.join(data_dir, "audio_segment_label")
    label_ctx_split_folder = os.path.join(data_dir, "audio_segment_label_ctx")
    trans_split_folder = os.path.join(data_dir, "audio_segment_transcript")

    # split_audio_word(data_dir, audio_name, label_name, label_split_folder)
    split_audio_word(data_dir, audio_name, trans_name, trans_split_folder)
    # split_audio_word_context(data_dir, audio_name, label_name, label_ctx_split_folder)

    return None


if __name__ == "__main__":
    main()
