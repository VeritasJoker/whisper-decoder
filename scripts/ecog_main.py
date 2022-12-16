import os
import numpy as np
import pandas as pd
import pickle
import whisper
from scipy.io import loadmat

from utils import load_label, write_config, save_pickle
from ecog_config import parse_arguments, setup_environ
from utils_whisper import brain_spectrogram
from audio_main import get_concat_labels


def ecog_to_spec(args, ecog):
    """Read audio file and create spectrogram"""
    ecog_padded = whisper.pad_or_trim(ecog, args.n_sample)
    ecog_spec = brain_spectrogram(ecog_padded, args.window_len, args.hop_len)

    assert ecog_spec.shape[0] == args.n_mel
    assert ecog_spec.shape[1] == args.n_frame

    return ecog_spec.numpy()


def select_elecs(args):
    sig_elec_file = os.path.join(args.data_dir, f"{args.sid}_elecs.csv")
    sig_elecs = pd.read_csv(sig_elec_file)

    if args.elec_type == "ifg":
        args.elecs = sig_elecs.ifg.dropna().astype(int).tolist()
    elif args.elec_type == "stg":
        args.elecs = sig_elecs.stg.dropna().astype(int).tolist()
    elif args.elec_type == "both":
        ifg_elecs = sig_elecs.ifg.dropna().astype(int).tolist()
        stg_elecs = sig_elecs.stg.dropna().astype(int).tolist()
        args.elecs = ifg_elecs + stg_elecs
    else:
        assert args.elec_type == "all"

    return args


def get_elec_data(args):

    ecogs = []

    for i in args.elecs:
        filename = (
            f"NY{args.sid}_111_Part1_conversation1_electrode_preprocess_file_{i}.mat"
        )

        elec_file = os.path.join(args.ecog_path, filename)
        try:
            e = loadmat(elec_file)["p1st"].squeeze().astype(np.float32)
            ecogs.append(e)
        except:
            print(f"No ecog data for elec {i}")
    ecogs = np.asarray(ecogs).T

    return ecogs


def get_chunk_data(df, i, args):

    # TODO: Sorted dataframe to check for overlap

    offset = df.brain_offset[i]  # current offset
    onset = np.max([df.iloc[0]["brain_onset"], offset - args.n_sample])  # temp onset

    # account for if a word is partially at the start of window
    onset_idx = df.brain_onset.ge(onset).idxmax()  # first word with onset inside window
    offset_idx = df.brain_offset.ge(
        onset
    ).idxmax()  # first word with offset inside window
    if onset_idx > offset_idx:  # if a word partially in window
        onset = df.loc[offset_idx, "brain_offset"]  # take its offset as onset

    onset = int(onset) + args.shift_fs
    offset = int(offset) + args.shift_fs
    assert offset - onset <= args.n_sample

    label = get_concat_labels(df, onset_idx, i)

    return (onset, offset, label)


def split_ecog(args):

    ecogs = get_elec_data(args)  # get brain ecog data
    df = load_label(args.datum_path)  # load datum
    df.reset_index(inplace=True)

    if args.seg_type != "word":  # new column to store labels
        words = []
    if args.seg_type == "sentence":
        prev_offset_idx = df.index[0]

    ecog_specs = []

    for i in df.index:
        if i % 10 == 0:
            print(i)

        if args.seg_type == "word":
            onset = int(df.brain_onset[i]) + args.shift_fs
            offset = int(df.brain_offset[i]) + args.shift_fs

        elif args.seg_type == "sentence":
            if df.word[i][-1] in "!?." or i == len(df.index) - 1:  # end of a sentence
                onset_idx, prev_offset_idx = prev_offset_idx + 1, i  # shift
                onset = int(df.brain_onset[onset_idx]) + args.shift_fs  # onset
                offset = int(df.brain_offset[i]) + args.shift_fs  # offset
                label = get_concat_labels(df, onset_idx, i)
                words.append(label)
                if offset - onset > args.n_sample:  # skip long sentence
                    print(label)
                    continue
            else:  # middle of a sentence
                continue

        elif args.seg_type == "chunk":
            onset, offset, label = get_chunk_data(df, i, args)
            words.append(label)

        if np.shape(ecogs)[0] < offset or onset < 0:  # out of bounds
            continue

        # calculate spectrogram
        ecog = ecogs[int(onset) : int(offset), :].mean(axis=1)  # take average
        ecog_specs.append(ecog_to_spec(args, ecog))

    # Less accurate but faster version than the for loop
    # df["brain_start"] = df.brain_onset.astype(int) - half_window + shift_fs
    # df["brain_end"] = df.brain_onset.astype(int) + half_window + shift_fs
    # df = df[(df.brain_start > 0) | (df.brain_end < ecogs.shape[0])]  # filter df

    # elec_data2 = [
    #     ecogs[start:end].mean(axis=1)
    #     for start, end in zip(df.brain_start, df.brain_end)
    # ]
    # elec_data2 = np.stack(elec_data2, axis=0)

    if args.seg_type == "word":
        words = df.word.tolist()

    assert len(words) == len(ecog_specs)
    result = {
        "ecog_specs": ecog_specs,
        "label": words,
        # "index": index,
        # "df_index": df_index,
    }
    pkl_dir = os.path.join(args.result_dir, f"{args.sid}_ecog_{args.elec_type}_spec")
    save_pickle(result, pkl_dir)
    return None


# def spec_ecog(args):
#     pickle_name = os.path.join(
#         args.result_dir, f"{args.sid}_ecog_{args.elec_type}_segment.pkl"
#     )
#     assert os.path.isfile(pickle_name), "No ecog data"
#     with open(pickle_name, "rb") as fh:
#         datum = pickle.load(fh)

#     ecogs = datum["ecog_segs"]
#     ecog_specs = []

#     for index, ecog in enumerate(ecogs):
#         print(index)
#         ecog_specs.append(ecog_to_spec(args, ecog))

#     assert len(ecog_specs) == len(ecogs)
#     result = {
#         "ecog_specs": ecog_specs,
#         "label": datum["label"],
#         "index": datum["index"],
#         "df_index": datum["df_index"],
#     }
#     pkl_dir = os.path.join(args.result_dir, f"{args.sid}_ecog_{args.elec_type}_spec")
#     save_pickle(result, pkl_dir)

#     return None


def main():
    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data and results
    args = setup_environ(args)

    # select significant electrodes
    args = select_elecs(args)

    # Saving configuration to output directory
    write_config(vars(args))

    split_ecog(args)

    return


if __name__ == "__main__":
    main()
