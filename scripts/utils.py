import pickle
import pandas as pd
import scipy.io.wavfile as wavfile


def load_transcript(filepath):
    df = pd.read_csv(filepath, header=None, sep=" ")
    df.rename(
        columns={
            0: "word",
            1: "onset",
            2: "offset",
            3: "unknown",
            4: "speaker",
        },
        inplace=True,
    )

    df["audio_onset"] = (df.onset + 3000) / 512
    df["audio_offset"] = (df.offset + 3000) / 512

    df = df.dropna(subset=["onset", "offset"])

    return df


def load_label(filepath):

    with open(filepath, "rb") as f:
        full_labels = pickle.load(f)
        labels_df = pd.DataFrame(full_labels["labels"])

    labels_df["onset_sec"] = (labels_df.onset + 3000) / 512
    labels_df["offset_sec"] = (labels_df.offset + 3000) / 512

    labels_df = labels_df.dropna(subset=["onset_sec", "offset_sec"])

    return labels_df


def load_audio(filepath):
    fs, audio = wavfile.read(filepath)
    print(f"Sampling rate: {fs}")
    print(f"Audio Length (s): {len(audio) / fs}")
    return fs, audio
