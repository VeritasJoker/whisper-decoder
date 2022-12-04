import os
from scipy.io import loadmat
import pickle
import pandas as pd
import numpy as np
import torch
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset

from utils import load_pickle
from model_inference import load_whisper_model
from model_config import parse_arguments, write_model_config


def model_tokenize(labels, tokenizer):

    labels = pd.DataFrame(labels)
    labels = labels.rename(columns={0:"label"})
    # labels["tokens"] = labels.label.apply(tokenizer.tokenize)
    labels["token_ids"] = labels.label.apply(tokenizer.encode)

    return labels.token_ids.tolist()



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
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
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [label.lower() for label in pred_str]
    label_str = [label.lower() for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# def prepare_dataset(batch):

#     batch["input_features"] = batch["ecog_specs"]
#     batch["labels"] = tokenizer(batch["label"]).input_ids

#     return batch


def get_trainer(args, data_all):

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

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
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
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
        tokenizer=processor.tokenizer,
    )

    return trainer, training_args




def main():

    # Read command line arguments
    args = parse_arguments()

    write_model_config(vars(args))

    global model, processor, tokenizer, metric  # global variables
    model, processor, tokenizer = load_whisper_model(args.model_size)

    # model.config.forced_decoder_ids = None  # not sure if we need these
    # model.config.suppress_tokens = []
    metric = evaluate.load("./metrics/wer")

    ecog_pkl = load_pickle(args.datafile)
    labels = model_tokenize(ecog_pkl["label"])
    ecog_data = {"input_features":ecog_pkl["ecog_specs"],"labels":labels}
    ecog_data = Dataset.from_dict(ecog_pkl)
    
    # Split train / test
    # 8/2 split: train_size = 4029, test_size = 1008
    # 9/1 split: train_size = 4533, test_size = 504
    # 9.5/0.5 split: train_size = 4785, test_size = 252
    data_all = ecog_data.train_test_split(test_size=args.data_split, shuffle=False)

    trainer, training_args = get_trainer(args, data_all)

    print("Saving processor")
    processor.save_pretrained(training_args.output_dir)

    print("Start training")
    trainer.train()

    return None


if __name__ == "__main__":
    main()
