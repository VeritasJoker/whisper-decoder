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


def res_evaluate(seg_type, result_type):
    # Set up
    results_dir = f"./results/{seg_type}"
    os.makedirs(results_dir, exist_ok=True)
    label_pkl = f"./seg-data/podcast/{seg_type}/{result_type}_spec.pkl"
    labels = load_pickle(label_pkl)

    files = glob.glob(f"{results_dir}/{result_type}_tiny.csv")

    metric1 = evaluate.load("./metrics/wer")
    metric2 = evaluate.load("./metrics/cer")

    li = []

    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=1, ignore_index=False)
    df["label"] = labels["label"]

    df = remove_punc_df(df)
    if seg_type == "word":
        df = get_single_pred_df(df)

    train_idx = round(len(df) * (1 - 0.1))  # train/test split
    df2 = df.loc[train_idx:, :]
    df2.reset_index(inplace=True)

    results_df = pd.DataFrame(columns=["wer", "cer"])
    for column in df.columns:
        if column != "label" and column != "index":
            column_eval = column + "-eval"
            results_df.loc[column, "wer"] = metric1.compute(
                predictions=df[column], references=df.label
            )
            results_df.loc[column_eval, "wer"] = metric1.compute(
                predictions=df2[column], references=df2.label
            )
            results_df.loc[column, "cer"] = metric2.compute(
                predictions=df[column], references=df.label
            )
            results_df.loc[column_eval, "cer"] = metric2.compute(
                predictions=df2[column], references=df2.label
            )
    results_df.to_csv(f"./results/{seg_type}/summary_{result_type}.csv")


def main():

    # Parameters
    seg_types = ["chunk", "sentence"]

    result_type = "audio"
    result_type = "717_ecog_ifg"
    result_types = [
        "742_ecog_ifg",
        "742_ecog_stg",
        "742_ecog_both",
        "742_ecog_all",
        "742_ecog_ifg_0",
        "742_ecog_stg_0",
        "742_ecog_both_0",
        "742_ecog_all_0",
        "798_ecog_ifg",
        "798_ecog_stg",
        "798_ecog_both",
        "798_ecog_all",
        "798_ecog_ifg_0",
        "798_ecog_stg_0",
        "798_ecog_both_0",
        "798_ecog_all_0",
    ]

    for seg_type in seg_types:
        for result_type in result_types:
            res_evaluate(seg_type, result_type)

    return


if __name__ == "__main__":
    main()
