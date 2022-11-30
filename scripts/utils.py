import os
import json
import pickle
import pandas as pd
import scipy.io.wavfile as wavfile


def load_pickle(filepath):
    print(f"Loading: {filepath}")
    with open(filepath, "rb") as fh:
        pkl = pickle.load(fh)
    return pkl


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'"""
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as fh:
        pickle.dump(item, fh)
    return


# def load_transcript(filepath):
#     df = pd.read_csv(filepath, header=None, sep=" ")
#     df.rename(
#         columns={
#             0: "word",
#             1: "onset",
#             2: "offset",
#             3: "unknown",
#             4: "speaker",
#         },
#         inplace=True,
#     )

#     df["audio_onset"] = (df.onset + 3000) / 512
#     df["audio_offset"] = (df.offset + 3000) / 512

#     df = df.dropna(subset=["onset", "offset"])

#     df = df.rename(columns={"onset": "brain_onset", "offset": "brain_offset"})

#     return df


def load_label(filepath):

    labels_df = pd.DataFrame(load_pickle(filepath)["labels"])

    labels_df["audio_onset"] = (labels_df.onset + 3000) / 512
    labels_df["audio_offset"] = (labels_df.offset + 3000) / 512

    labels_df = labels_df.dropna(subset=["audio_onset", "audio_offset"])

    labels_df = labels_df.rename(
        columns={"onset": "brain_onset", "offset": "brain_offset"}
    )

    return labels_df


def load_audio(filepath):
    fs, audio = wavfile.read(filepath)
    print(f"Sampling rate: {fs}")
    print(f"Audio Length (s): {len(audio) / fs}")
    return fs, audio


def write_config(dictionary):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    if "audio_path" in dictionary.keys():
        config_type = "audio"
    elif "ecog_path" in dictionary.keys():
        config_type = "ecog"

    config_file = os.path.join(dictionary["result_dir"], f"{config_type}_config.json")
    with open(config_file, "w") as outfile:
        outfile.write(json_object)
