import os
import numpy as np
import pandas as pd
import pickle
import whisper
from scipy.io import loadmat

from utils import load_label, write_config, save_pickle
from ecog_config import parse_arguments, setup_environ
from utils_whisper import brain_spectrogram


def ecog_to_spec(args, ecog):
    """Read audio file and create spectrogram"""
    ecog_padded = whisper.pad_or_trim(ecog, args.n_sample)
    ecog_spec = brain_spectrogram(ecog_padded, args.hop_len)

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


def split_ecog(args):

    ecogs = get_elec_data(args)  # get brain ecog data
    df = load_label(args.datum_path)  # load datum

    total_word_len = len(df)
    elec_data = np.zeros((total_word_len, args.window_fs))

    if "spec" in args.save_type:  # list to store audio specs
        ecog_specs = []

    df_index = df.index.tolist()
    df.reset_index(inplace=True)
    for i in df.index:
        print(i)
        if args.seg_type == "word":
            onset = int(df.brain_onset[i])
            # offset = df.brain_offset[i]

            start = (onset) - args.half_window + args.shift_fs
            end = (onset) + args.half_window + args.shift_fs

        if np.shape(ecogs)[0] < end or start < 0:  # out of bounds
            continue

        ecog = ecogs[int(start) : int(end), :].mean(axis=1)  # take average
        elec_data[i, :] = ecog  # save segment data
        if "spec" in args.save_type:
            # calculate spectrogram
            ecog_specs.append(ecog_to_spec(args, elec_data[i, :]))

    # Less accurate but faster version than the for loop # -7.290375709533691
    # df["brain_start"] = df.brain_onset.astype(int) - half_window + shift_fs
    # df["brain_end"] = df.brain_onset.astype(int) + half_window + shift_fs
    # df = df[(df.brain_start > 0) | (df.brain_end < ecogs.shape[0])]  # filter df

    # elec_data2 = [
    #     ecogs[start:end].mean(axis=1)
    #     for start, end in zip(df.brain_start, df.brain_end)
    # ]
    # elec_data2 = np.stack(elec_data2, axis=0)

    words = df.word.tolist()
    index = df.index.tolist()
    assert len(words) == len(df_index) == len(index) == len(elec_data)
    if "ecog" in args.save_type:
        result = {
            "ecog_segs": elec_data,
            "label": words,
            "index": index,
            "df_index": df_index,
        }
        pkl_dir = os.path.join(
            args.result_dir, f"{args.sid}_ecog_{args.elec_type}_segment"
        )
        save_pickle(result, pkl_dir)
    if "spec" in args.save_type:
        assert len(words) == len(ecog_specs)
        result = {
            "ecog_specs": ecog_specs,
            "label": words,
            "index": index,
            "df_index": df_index,
        }
        pkl_dir = os.path.join(
            args.result_dir, f"{args.sid}_ecog_{args.elec_type}_spec"
        )
        save_pickle(result, pkl_dir)
    return None


def spec_ecog(args):
    pickle_name = os.path.join(
        args.result_dir, f"{args.sid}_ecog_{args.elec_type}_segment.pkl"
    )
    assert os.path.isfile(pickle_name), "No ecog data"
    with open(pickle_name, "rb") as fh:
        datum = pickle.load(fh)

    ecogs = datum["ecog_segs"]
    ecog_specs = []

    for index, ecog in enumerate(ecogs):
        print(index)
        ecog_specs.append(ecog_to_spec(args, ecog))

    assert len(ecog_specs) == len(ecogs)
    result = {
        "ecog_specs": ecog_specs,
        "label": datum["label"],
        "index": datum["index"],
        "df_index": datum["df_index"],
    }
    pkl_dir = os.path.join(args.result_dir, f"{args.sid}_ecog_{args.elec_type}_spec")
    save_pickle(result, pkl_dir)

    return None


def main():
    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data and results
    args = setup_environ(args)

    # select significant electrodes
    args = select_elecs(args)

    # Saving configuration to output directory
    write_config(vars(args))

    if args.save_type == "spec":
        # just do spectrogram
        spec_ecog(args)
    else:
        # split ecog data
        split_ecog(args)

    return


if __name__ == "__main__":
    main()
