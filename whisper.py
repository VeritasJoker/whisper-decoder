import os
import torch
import pickle
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperFeatureExtractor,
)
import pandas as pd
import numpy as np
import math
from pydub import AudioSegment


def load_whisper_mode():

    modelname = "tiny"

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-" + modelname + ".en"
    )
    model = WhisperModel.from_pretrained(
        "openai/whisper-" + modelname + ".en",
        output_hidden_states=True,
        return_dict=True,
    )


def load_transcript(filepath):
    df = pd.read_csv(filepath, header=None, sep=" ")
    df.rename(
        columns={
            0: "word",
            1: "onset",
            2: "offset",
            3: "unknown",
            4: "speaker",
        },
        inplace=True,
    )

    df["audio_onset"] = df.onset / 512
    df["audio_offset"] = df.offset / 512

    return df


def load_audio(filepath):
    audio = AudioSegment.from_file(filepath)

    return audio


def main():
    audio = load_audio("data/Podcast.wav")
    df = load_transcript("data/podcastAlignedDatum.txt")

    return None


if __name__ == "__main__":
    main()
