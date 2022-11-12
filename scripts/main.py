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


def load_whisper_model(model_size):

    processor = WhisperProcessor.from_pretrained("openai/whisper-" + model_size + ".en")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-" + model_size + ".en",
        output_hidden_states=True,
        return_dict=True,
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-" + model_size + ".en",
    )

    return model, processor, tokenizer


def transcribe(filename):

    model1, processor1, tokenizer1 = load_whisper_model("tiny")
    model2, processor2, tokenizer2 = load_whisper_model("base")
    breakpoint()

    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model2.device)
    input_features = mel.unsqueeze(dim=0)

    decoder_input_ids = torch.tensor([[1, 1]]) * model2.config.decoder_start_token_id
    output = model2(input_features, decoder_input_ids=decoder_input_ids)
    # print(output.keys())

    # print(len(output.decoder_hidden_states))

    # for i in output.decoder_hidden_states:
    #     print(i.shape)

    generated_ids = model2.generate(inputs=input_features)
    breakpoint()

    transcription = processor2.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)

    return None


def main():
    project = "podcast"
    data_dir = os.path.join("data", project, "audio_segment_label")

    sample = "segment_5099-characteristic.wav"
    transcribe(os.path.join(data_dir, sample))

    return None


if __name__ == "__main__":
    main()
