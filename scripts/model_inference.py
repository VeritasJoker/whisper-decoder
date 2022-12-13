import os
import torch
import pickle
import numpy as np
import pandas as pd
import whisper
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)

from utils import load_pickle, load_label


def load_whisper_model(model_size):

    model_fullname = f"openai/whisper-{model_size}.en"
    if model_size == "large":
        model_fullname = f"openai/whisper-large"
    print(f"Loading {model_fullname}")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_fullname,
        output_hidden_states=True,
        return_dict=True,
    )
    processor = WhisperProcessor.from_pretrained(model_fullname)
    tokenizer = WhisperTokenizer.from_pretrained(model_fullname)

    return model, processor, tokenizer


def load_whisper_model_by_path(model_path, checkpoint):

    processor = WhisperProcessor.from_pretrained(model_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_path)

    model_path = os.path.join(model_path, f"checkpoint-{checkpoint}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        output_hidden_states=True,
        return_dict=True,
    )

    return model, processor, tokenizer


def transcribe_audio(model, processor, filename):

    # load and prepare audio
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    input_features = mel.unsqueeze(dim=0)

    # model generate (greedy decoding)
    output = model.generate(inputs=input_features, max_new_tokens=448)
    transcription = processor.batch_decode(output, skip_special_tokens=True)[0]

    return transcription


def transcribe_spec(model, processor, spec):

    # prepare spec
    spec = torch.from_numpy(spec)
    input_features = spec.unsqueeze(dim=0)

    # model generate (greedy decoding)
    output = model.generate(inputs=input_features, max_new_tokens=448)
    transcription = processor.batch_decode(output, skip_special_tokens=True)[0]

    return transcription


def transcribe_decoder():

    # just a place to put spare code
    # decoder_input_ids = (
    #     torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    # )

    # output = model2(input_features, decoder_input_ids=decoder_input_ids)
    # breakpoint()
    # print(output.keys())

    # print(len(output.decoder_hidden_states))

    # for i in output.decoder_hidden_states:
    #     print(i.shape)

    # output = model.generate(inputs=input_features, return_dict_in_generate=True)

    return None


def main():

    project = "tfs"
    data_dir = os.path.join("data", project)
    sample = "798_30s_test.wav"

    project = "podcast"
    data_dir = os.path.join("seg-data", project, "word", "audio_segment")
    sample = "segment_5096-a.wav"

    # project = "podcast"
    # data_dir = os.path.join("data", project)
    # sample = "podcast_segment_5099-characteristic.wav"

    model, processor, _ = load_whisper_model("tiny")
    breakpoint()
    result = transcribe_audio(model, processor, os.path.join(data_dir, sample))
    print(result)

    return None


if __name__ == "__main__":
    main()
