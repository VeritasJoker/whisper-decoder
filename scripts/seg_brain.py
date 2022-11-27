import os
import pickle
import numpy as np
import pandas as pd
import csv
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from functools import reduce
from utils import load_label


def get_elec_data(subject, brain_data_dir, electrodes):

    ecogs = []

    for i in electrodes:
        filename = f"NY{subject}_111_Part1_conversation1_electrode_preprocess_file_{i}.mat"

        elec_file = os.path.join(brain_data_dir, filename)

        e = loadmat(elec_file)["p1st"].squeeze().astype(np.float32)
        ecogs.append(e)
    ecogs = np.asarray(ecogs)

    return ecogs


def seg_data(subject, df, result_dir, ecogs, ecogs_ifg, ecogs_stg):
    # bin_ms = 62.5
    shift_ms = 300  # onset shift
    window_ms = 625  # window size
    fs = 512  # s to hz

    # bin_fs = int(bin_ms / 1000 * fs)
    shift_fs = int(shift_ms / 1000 * fs)
    window_fs = int(window_ms / 1000 * fs)
    half_window = window_fs // 2
    # n_bins = window_ms // bin_ms
    # n_bins = window_fs // bin_fs

    total_word_len = len(df)
    # total_word_len = 5884

    datum = []
    word_number = []
    elec_data = np.zeros((total_word_len, window_fs))
    elec_data_ifg = np.zeros((total_word_len, window_fs))
    elec_data_stg = np.zeros((total_word_len, window_fs))

    # df["brain_start"] = df.brain_onset.astype(int) - half_window + shift_fs
    # df["brain_end"] = df.brain_onset.astype(int) + half_window + shift_fs
    # df = df[(df.brain_start > 0) | (df.brain_end < ecogs.shape[0])]  # filter df

    # elec_data2 = [
    #     ecogs[start:end].mean(axis=1)
    #     for start, end in zip(df.brain_start, df.brain_end)
    # ]
    # elec_data2 = np.stack(elec_data2, axis=0)

    df.reset_index(inplace=True)
    for i in df.index:
        onset = int(df.brain_onset[i])
        # offset = df.brain_offset[i]
        datum.append(df.word[i])
        word_number.append(i)

        start = (onset) - half_window + shift_fs
        end = (onset) + half_window + shift_fs

        if np.shape(ecogs)[0] < end or start < 0:  # out of bounds
            continue

        w = ecogs[int(start) : int(end), :].mean(axis=1)
        w_ifg = ecogs_ifg[int(start) : int(end), :].mean(axis=1)
        w_stg = ecogs_stg[int(start) : int(end), :].mean(axis=1)

        elec_data[i, :] = w
        elec_data_ifg[i, :] = w_ifg
        elec_data_stg[i, :] = w_stg

    savemat(
        os.path.join(result_dir, f"{subject}_brain_data.mat"),
        {
            "ifg_data": elec_data_ifg,
            "stg_data": elec_data_stg,
            "brain_data_both": elec_data,
            "datum": datum,
            "word_number": word_number,
        },
    )
    return None


def main():

    # meta arguments
    project = "podcast"
    label_name = "777_full_labels.pkl"

    # define data and result directories
    data_dir = os.path.join("data", project)
    result_dir = os.path.join("seg-data", project, "brain")

    subject = 717
    brain_data_dir = f"/projects/HASSON/247/data/podcast/NY{subject}_111_Part1_conversation1/preprocessed"

    # significant electrodes
    ifg_sig = [
        4,
        9,
        10,
        18,
        27,
        66,
        71,
        74,
        75,
        78,
        79,
        80,
        82,
        86,
        87,
        88,
        95,
        108,
    ]
    stg_sig = [
        36,
        37,
        38,
        39,
        46,
        47,
        112,
        113,
        114,
        116,
        117,
        119,
        120,
        121,
        122,
        126,
    ]

    electrodes = ifg_sig + stg_sig

    ecogs = get_elec_data(subject, brain_data_dir, electrodes)

    ecogs_ifg = ecogs[0 : len(ifg_sig), :]
    ecogs_stg = ecogs[len(ifg_sig) : len(stg_sig) + len(ifg_sig), :]

    ecogs = np.asarray(ecogs).T
    ecogs_ifg = np.asarray(ecogs_ifg).T
    ecogs_stg = np.asarray(ecogs_stg).T

    df = load_label(os.path.join(data_dir, label_name))

    seg_data(subject, df, result_dir, ecogs, ecogs_ifg, ecogs_stg)


if __name__ == "__main__":
    main()
