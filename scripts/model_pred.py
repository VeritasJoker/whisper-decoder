import os
import pickle
import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from utils import load_pickle
from model_inference import load_whisper_model
from model_config import parse_arguments, write_model_config
from model_train import model_tokenize, DataCollatorSpeechSeq2SeqWithPadding


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


def get_trainer_for_eval(args, data_all):

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

    global model, processor, tokenizer, metric  # global variables
    model, processor, tokenizer = load_whisper_model(args.model_size)
    metric = evaluate.load("./metrics/wer")

    print("Loading data")
    data_pkl = load_pickle(os.path.join(args.data_dir, args.eval_file))
    labels = model_tokenize(data_pkl["label"], tokenizer)

    if "audio" in args.eval_file:
        data_dict = {"input_features":data_pkl["audio_specs"],"labels":labels}
    else:
        data_dict = {"input_features":data_pkl["ecog_specs"],"labels":labels.token_ids.tolist()}
    
    data = Dataset.from_dict(data_dict)
    data_all = data.train_test_split(test_size=args.data_split, shuffle=False)
    
    trainer, _ = get_trainer_for_eval(args, data_all)

    print("Doing predictions")
    preds = trainer.predict(test_dataset=data)
    results = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)

    results = pd.DataFrame(results, columns = [f"{args.model_size}_preds"])

    results.to_csv(f"results/{args.eval_file[:-9]}_{args.model_size}.csv", index=False)

    return None


if __name__ == "__main__":
    main()
