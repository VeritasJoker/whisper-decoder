import os
from scipy.io import loadmat
import pickle
import pandas as pd

from utils import load_pickle
from model_inference import (
    transcribe_spec,
    load_whisper_model,
    load_whisper_model_by_path,
)


def main():

    project = "podcast"
    data_dir = os.path.join("seg-data", project, "word")

    # ecog_ifg = load_pickle(os.path.join(data_dir, "717_ecog_ifg_spec.pkl"))
    # ecog_stg = load_pickle(os.path.join(data_dir, "717_ecog_stg_spec.pkl"))
    # ecog_both = load_pickle(os.path.join(data_dir, "717_ecog_both_spec.pkl"))
    # ecog_all = load_pickle(os.path.join(data_dir, "717_ecog_all_spec.pkl"))
    audio = load_pickle(os.path.join(data_dir, "audio_spec.pkl"))

    breakpoint()

    model, processor, _ = load_whisper_model("tiny")
    # model.save_pretrained("./whisper-decoder")

    breakpoint()
    model2 = load_whisper_model_by_path("./whisper-decoder")

    word = transcribe_spec(model2, processor, audio["audio_specs"][1])
    print(word)
    # df = pd.DataFrame.from_dict(datum)
    return None


if __name__ == "__main__":
    main()
