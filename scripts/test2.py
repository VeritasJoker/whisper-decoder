import os
import pickle
import pandas as pd
import evaluate

from utils import load_pickle
from model_inference import (
    transcribe_spec,
    load_whisper_model,
    load_whisper_model_by_path
)
from model_config import parse_arguments



def main():

    # Read command line arguments
    args = parse_arguments()

    # model.save_pretrained("./whisper-decoder")
    metric = evaluate.load("./metrics/wer")
    checkpoint = 1000
    checkpoint = 2000
    model, processor, _ = load_whisper_model_by_path(f"./models/{args.saving_dir}", checkpoint)

    ecog = load_pickle(args.datafile)

    word = ecog["label"][0]
    pred_word = transcribe_spec(model, processor, ecog["ecog_specs"][0])

    breakpoint()

    return


if __name__ == "__main__":
    main()