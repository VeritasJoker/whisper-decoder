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


def main():
    ifg_significant = [
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
    stg_significant = [
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

    electrodes = [ifg_significant, stg_significant]
    electrodes = reduce(lambda a, b: a + b, electrodes)

    subject = 717
    ecogs = []
    path_transcript = ""
    path_elec = ""

    for i in electrodes:
        filename = (
            "NY"
            + str(subject)
            + "_111_Part1_conversation1_electrode_preprocess_file_"
            + str(i)
            + ".mat"
        )

        path_elec_file = path_elec + "\\" + filename

        e = loadmat(path_elec_file)["p1st"].squeeze().astype(np.float32)
        ecogs.append(e)
    ecogs = np.asarray(ecogs)

    ecogs_ifg = ecogs[0 : len(ifg_significant), :]
    ecogs_stg = ecogs[
        len(ifg_significant) : len(stg_significant) + len(ifg_significant), :
    ]

    ecogs = np.asarray(ecogs).T
    ecogs_ifg = np.asarray(ecogs_ifg).T
    ecogs_stg = np.asarray(ecogs_stg).T

    df = load_label(os.path.join(path_transcript, "777_full_labels.pkl"))

    bin_ms = 62.5
    shift_ms = 300
    window_ms = 625
    fs = 512

    bin_fs = int(bin_ms / 1000 * fs)
    shift_fs = int(shift_ms / 1000 * fs)
    window_fs = int(window_ms / 1000 * fs)
    half_window = window_fs // 2
    # n_bins = window_ms // bin_ms
    n_bins = window_fs // bin_fs

    # total_word_len=np.shape(df.index)
    total_word_len = 5884

    elec_data = []
    elec_data_ifg = []
    elec_data_stg = []
    datum = []
    word_number = []

    elec_data = np.zeros((total_word_len, 320))
    elec_data_ifg = np.zeros((total_word_len, 320))
    elec_data_stg = np.zeros((total_word_len, 320))

    for i in df.index:

        onset = int(df.brain_onset[i])
        # offset = df.brain_offset[i]
        datum.append(df.word[i])
        word_number.append(i)

        start = (onset) - half_window + shift_fs
        end = (onset) + half_window + shift_fs

        if np.shape(ecogs)[0] < end or start < 0:
            breakpoint()
            # FIXME

        w = ecogs[int(start) : int(end), :].mean(axis=1)
        w_ifg = ecogs_ifg[int(start) : int(end), :].mean(axis=1)
        w_stg = ecogs_stg[int(start) : int(end), :].mean(axis=1)

        # elec_data.append(w)
        # elec_data_ifg.append(w_ifg)
        # elec_data_stg.append(w_stg)

        elec_data[i, :] = w
        elec_data_ifg[i, :] = w_ifg
        elec_data_stg[i, :] = w_stg

    # savemat('whisper_brain_data.mat',{'ifg_data':elec_data_ifg,'stg_data':elec_data_stg,'brain_data_both':elec_data,'datum':datum, 'word_number':word_number})
