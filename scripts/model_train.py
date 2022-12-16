import os
from scipy.io import loadmat
import pickle
import pandas as pd
import numpy as np
import torch
import string
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, DatasetDict

from utils import load_pickle
from model_inference import load_whisper_model
from model_config import parse_arguments, write_model_config


def model_tokenize_word(labels, tokenizer):

    labels = pd.DataFrame(labels)
    labels = labels.rename(columns={0: "label"})
    labels["label_lower"] = labels.label.str.lower()  # lower case
    labels["token_ids"] = labels.label_lower.apply(tokenizer.encode)  # tokenize

    puncs = []
    for punc in string.punctuation:  # 1-by-1 (weird interactions if not)
        puncs = puncs + tokenizer.encode(punc)[2:-1]  # tokenize punctuation

    labels["token_ids_nopunc"] = labels.token_ids.apply(
        lambda x: [token for token in x if token not in puncs]
    )  # get rid of punctuations

    return labels.token_ids_nopunc.tolist()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (
            (labels[:, 0] == self.processor.tokenizer.bos_token_id)
            .all()
            .cpu()
            .item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def remove_punc(input_str):
    for character in string.punctuation:  # remove punc
        input_str = input_str.replace(character, "")
    return input_str.lower()  # lowercase


def get_first_word(sentence):

    if len(sentence) > 0:
        first_word = sentence.split()[0]  # get first word
    else:
        return sentence  # empty prediction
    first_word = remove_punc(first_word)

    return first_word


def compute_metric(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [remove_punc(pred) for pred in pred_str]
    label_str = [remove_punc(label) for label in label_str]

    metric_num = metric.compute(predictions=pred_str, references=label_str)

    return {"metric": metric_num}


def compute_metric_word(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # make it word level, remove punc, and lowercase
    pred_str = [get_first_word(pred) for pred in pred_str]
    label_str = [get_first_word(label) for label in label_str]

    metric_num = metric.compute(predictions=pred_str, references=label_str)

    return {"metric": metric_num}


# def prepare_dataset(batch):

#     batch["input_features"] = batch["ecog_specs"]
#     batch["labels"] = tokenizer(batch["label"]).input_ids

#     return batch


def get_trainer(args, data_all):

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    if args.seg_type == "word":
        metric_func = compute_metric_word
    else:
        metric_func = compute_metric

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./models/{args.saving_dir}",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="no",
        # save_steps=100,
        eval_steps=100,
        logging_steps=50,
        # load_best_model_at_end=True,
        # metric_for_best_model="metric",
        # greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_all["train"],
        eval_dataset=data_all["test"],
        data_collator=data_collator,
        compute_metrics=metric_func,
        tokenizer=processor.tokenizer,
    )

    return trainer, training_args


def prepare_data(args):

    print("Prepare data")

    all_specs_train = []
    all_specs_test = []
    all_labels_train = []
    all_labels_test = []

    for pkl in args.datafiles:
        ecog_pkl = load_pickle(pkl)
        labels = model_tokenize_word(ecog_pkl["label"], tokenizer)

        train_idx = round(
            len(labels) * (1 - args.data_split)
        )  # train/test split
        if (
            args.data_split_type == "train2.7-test0.3"
        ):  # train/test on all 3 patients
            all_specs_train = (
                all_specs_train + ecog_pkl["ecog_specs"][0:train_idx]
            )
            all_specs_test = all_specs_test + ecog_pkl["ecog_specs"][train_idx:]
            all_labels_train = all_labels_train + labels[0:train_idx]
            all_labels_test = all_labels_test + labels[train_idx:]
        elif (
            args.data_split_type == "train0.9-test0.1"
        ):  # train/test on 1 patient
            if pkl == args.testfile:
                all_specs_train = (
                    all_specs_train + ecog_pkl["ecog_specs"][0:train_idx]
                )
                all_specs_test = (
                    all_specs_test + ecog_pkl["ecog_specs"][train_idx:]
                )
                all_labels_train = all_labels_train + labels[0:train_idx]
                all_labels_test = all_labels_test + labels[train_idx:]
        elif (
            args.data_split_type == "train2.9-test0.1"
        ):  # train on 2.9, test 0.1
            if pkl == args.testfile:
                all_specs_train = (
                    all_specs_train + ecog_pkl["ecog_specs"][0:train_idx]
                )
                all_specs_test = (
                    all_specs_test + ecog_pkl["ecog_specs"][train_idx:]
                )
                all_labels_train = all_labels_train + labels[0:train_idx]
                all_labels_test = all_labels_test + labels[train_idx:]
            else:
                all_specs_train = all_specs_train + ecog_pkl["ecog_specs"]
                all_labels_train = all_labels_train + labels
        elif (
            args.data_split_type == "train2-test0.1"
        ):  # train on 2, test on 0.1
            if pkl == args.testfile:
                all_specs_test = (
                    all_specs_test + ecog_pkl["ecog_specs"][train_idx:]
                )
                all_labels_test = all_labels_test + labels[train_idx:]
            else:
                all_specs_train = all_specs_train + ecog_pkl["ecog_specs"]
                all_labels_train = all_labels_train + labels
        elif (
            args.data_split_type == "train2-test1"
        ):  # train on 2 patients, test on 1
            if pkl == args.testfile:
                all_specs_test = all_specs_test + ecog_pkl["ecog_specs"]
                all_labels_test = all_labels_test + labels
            else:
                all_specs_train = all_specs_train + ecog_pkl["ecog_specs"]
                all_labels_train = all_labels_train + labels

    ecog_data_train = {
        "input_features": all_specs_train,
        "labels": all_labels_train,
    }
    ecog_data_test = {
        "input_features": all_specs_test,
        "labels": all_labels_test,
    }
    ecog_data_train = Dataset.from_dict(ecog_data_train)
    ecog_data_test = Dataset.from_dict(ecog_data_test)
    data_all = DatasetDict({"train": ecog_data_train, "test": ecog_data_test})

    print(data_all)

    return data_all


def main():

    # Read command line arguments
    args = parse_arguments()

    write_model_config(vars(args))

    print("Load model / metric")
    global model, processor, tokenizer, metric  # global variables
    model, processor, tokenizer = load_whisper_model(args.model_size)

    # model.config.forced_decoder_ids = None  # not sure if we need these
    model.config.suppress_tokens = []
    if args.seg_type == "word":
        metric = evaluate.load("./metrics/cer")
    else:
        metric = evaluate.load("./metrics/wer")

    data_all = prepare_data(args)

    trainer, training_args = get_trainer(args, data_all)

    print("Saving processor")
    processor.save_pretrained(training_args.output_dir)

    print("Start training")
    trainer.train()

    return None


if __name__ == "__main__":
    main()
