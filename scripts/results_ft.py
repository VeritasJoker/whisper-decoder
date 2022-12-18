import os
import pickle
import pandas as pd
import evaluate
import glob
import string


def summarize_logs():

    files = glob.glob(f"logs/*.log")
    result_dir = "logs-summary/"

    name_str = "--saving-dir"
    eval_str = "eval_loss"
    save_str = "5000/5000"
    save_str2  = "CANCELLED AT"

    for file in files:
        with open(file) as f:
            f = f.readlines()
        results_df = pd.DataFrame(columns=["loss", "wer", "cer", "first_cer"])
        saving = False
        for line in f:
            if name_str in line:
                csvfilename = result_dir + line.split()[-1] + ".csv"
            if eval_str in line:
                epoch = line[line.find("epoch") + 8 : -2]
                results_df.loc[epoch, "loss"] = line[
                    line.find("eval_loss") + 12 : line.find("eval_wer") - 3
                ]
                results_df.loc[epoch, "wer"] = line[
                    line.find("eval_wer") + 11 : line.find("eval_cer") - 3
                ]
                results_df.loc[epoch, "cer"] = line[
                    line.find("eval_cer") + 11 : line.find("eval_fisrt_cer") - 3
                ]
                results_df.loc[epoch, "first_cer"] = line[
                    line.find("eval_fisrt_cer") + 17 : line.find("eval_runtime") - 3
                ]
            if save_str in line  or save_str2 in line:
                saving = True
        if len(results_df) == 0:
            saving = False
        if saving:
            results_df.reset_index(inplace=True)
            results_df.to_csv(csvfilename, index=False)

    return


def summarize_results():

    files = glob.glob(f"logs-summary/*.csv")
    result_dir = "results/ft/"

    for file in files:
        log_sum = pd.read_csv(file)
        log_sum.rename(columns={"index": "epoch"}, inplace=True)

        result = pd.DataFrame(columns=["epoch", "loss", "wer", "cer", "first_cer"])

        metrics = ["loss", "wer", "cer", "first_cer"]

        for metric in metrics:
            result.loc[metric, :] = log_sum.sort_values(
                by=[metric], ascending=True
            ).iloc[0]

        filename = result_dir + file.split("/")[-1]
        result.to_csv(filename, index=True)

    return


def main():

    summarize_logs()
    summarize_results()

    return


if __name__ == "__main__":
    main()
