import os
from scipy.io import loadmat
import pickle
import pandas as pd

from utils import load_pickle


def main():

    project = "podcast"
    data_dir = os.path.join("seg-data", project, "word")

    ecog = load_pickle(os.path.join(data_dir, "717_ecog_ifg_spec.pkl"))
    audio = load_pickle(os.path.join(data_dir, "audio_spec.pkl"))

    # df = pd.DataFrame.from_dict(datum)
    return None


if __name__ == "__main__":
    main()
