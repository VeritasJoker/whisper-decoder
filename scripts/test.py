import os
import pickle
import pandas as pd
import whisper
import torch

from utils import load_pickle
from model_inference import load_whisper_model, load_whisper_model_by_path


def main():

    project = "podcast"
    data_dir = os.path.join("seg-data", project, "chunk", "audio_segment")
    sample = "segment_5097-uniquely.wav"

    # project = "podcast"
    # data_dir = os.path.join("data", project)
    # sample = "podcast_segment_5099-characteristic.wav"

    model, processor, tokenizer = load_whisper_model("medium")

    audio = whisper.load_audio(os.path.join(data_dir, sample))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    input_features = mel.unsqueeze(dim=0)

    decoder_input_ids = torch.tensor([[1]]) * model.config.decoder_start_token_id

    output = model.generate(
        inputs=input_features,
        max_new_tokens=448,
        return_dict_in_generate=True,
        output_scores=True,
        forced_decoder_ids=[[0, 50257]],
    )
    timestamp_start = tokenizer.all_special_ids[-1] + 1

    transcription = ""
    for token in output.sequences[0]:
        if token >= timestamp_start:
            transcription = (
                transcription + f"<|{(token - timestamp_start) * 0.02:.2f}|>"
            )
        else:
            transcription = transcription + tokenizer.decode(token)

    breakpoint()
    # output2 = model(input_features, decoder_input_ids=decoder_input_ids)

    breakpoint()

    return


if __name__ == "__main__":
    main()
