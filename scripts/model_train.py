import os
from scipy.io import loadmat
import pickle
import pandas as pd
import torch
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, DatasetDict

from utils import load_pickle
from model_inference import (
    transcribe_spec,
    load_whisper_model,
    load_whisper_model_by_path,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["ecog_specs"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True).lower()
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True).lower()

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def prepare_dataset(batch):

    batch["input_features"] = batch["ecog_specs"]
    batch["labels"] = tokenizer(batch["label"]).input_ids

    return batch


def main():

    global model, processor, tokenizer, metric  # global variables
    model, processor, tokenizer = load_whisper_model("tiny")

    model.config.forced_decoder_ids = None  # not sure if we need these
    model.config.suppress_tokens = []
    metric = evaluate.load("./metrics/wer")

    project = "podcast"
    data_dir = os.path.join("seg-data", project, "word")

    ecog_ifg = load_pickle(os.path.join(data_dir, "717_ecog_ifg_spec.pkl"))
    # ecog_stg = load_pickle(os.path.join(data_dir, "717_ecog_stg_spec.pkl"))
    # ecog_both = load_pickle(os.path.join(data_dir, "717_ecog_both_spec.pkl"))
    # ecog_all = load_pickle(os.path.join(data_dir, "717_ecog_all_spec.pkl"))
    # audio = load_pickle(os.path.join(data_dir, "audio_spec.pkl"))

    data_all = DatasetDict()
    train_size = 4800

    train_ecog_ifg = {}
    test_ecog_ifg = {}

    for key in ecog_ifg.keys():
        train_ecog_ifg[key] = ecog_ifg[key][0:train_size]
        test_ecog_ifg[key] = ecog_ifg[key][train_size:]

    data_all["train"] = Dataset.from_dict(train_ecog_ifg)
    data_all["test"] = Dataset.from_dict(test_ecog_ifg)

    data_all = data_all.map(
        prepare_dataset, remove_columns=data_all.column_names["train"], num_proc=4
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-tiny-717-ifg-raw",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_all["train"],
        eval_dataset=data_all["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    return None


if __name__ == "__main__":
    main()
