import os
import pickle
import pandas as pd
import evaluate
import glob
import string

from utils import load_pickle


def remove_punc_df(df):
    for column in df.columns:
        df[column] = df[column].str.replace(r"[^\w\s]+", "", regex=True).str.lower()
    return df


def get_single_pred_df(df):

    for column in df.columns:
        if "pred" in column:
            df[f"{column}_word"] = df[column].str.split().str.get(0)

    df = df.fillna("")
    return df


def main():

    metric1 = evaluate.load("./metrics/wer")
    metric2 = evaluate.load("./metrics/cer")

    results_dir = "./results/words"
    label_pkl = "./seg-data/podcast/word/audio_spec.pkl"
    labels = load_pickle(label_pkl)

    result_type = "717_ecog_all"
    result_type = "audio"

    files = glob.glob(f"{results_dir}/{result_type}*.csv")

    li = []

    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=1, ignore_index=False)
    df["label"] = labels["label"]
    breakpoint()
    df = remove_punc_df(df)
    df = get_single_pred_df(df)

    results_df = pd.DataFrame(columns=["wer", "cer"])

    for column in df.columns:
        if column != "label":
            results_df.loc[column, "wer"] = metric1.compute(
                predictions=df[column], references=df.label
            )
            results_df.loc[column, "cer"] = metric2.compute(
                predictions=df[column], references=df.label
            )
    results_df.to_csv(f"./results/words/summary_{result_type}.csv")

    return


if __name__ == "__main__":
    main()
