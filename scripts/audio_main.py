import os
import numpy as np
import whisper
import scipy.io.wavfile as wavfile
from utils import load_label, load_audio, save_pickle, write_config
from audio_config import parse_arguments, setup_environ


def audio_to_spec(args, audio_file):
    """Read audio file and create spectrogram"""
    audio = whisper.load_audio(audio_file, args.sample_rate)
    audio_padded = whisper.pad_or_trim(audio, args.n_sample)
    audio_spec = whisper.log_mel_spectrogram(audio_padded)

    assert audio_spec.shape[0] == args.n_mel
    assert audio_spec.shape[1] == args.n_frame

    return audio_spec.numpy()


def get_chunk_data(df, i, onsetc, offsetc):

    offset = df[offsetc][i]  # current offset
    onset = np.max([df.iloc[0][onsetc], offset - 30])  # temp onset

    # account for if a word is partially at the start of window
    onset_idx = df[onsetc].ge(onset).idxmax()  # first word with onset inside window
    offset_idx = df[offsetc].ge(onset).idxmax()  # first word with offset inside window
    if onset_idx > offset_idx:  # if a word partially in window
        onset = df.loc[offset_idx, offsetc]  # take its offset as onset
    label = " ".join(
        df.loc[onset_idx:i, "word"].tolist()
    )  # first word with onset to current word

    return (onset, offset, label)


def split_audio(args):

    fs, full_audio = load_audio(args.audio_path)  # load audio file
    df = load_label(args.datum_path)  # load datum

    if args.seg_type != "word":  # new column to store labels
        words = []
    if args.seg_type == "sentence":
        prev_offset = df.iloc[0]["audio_onset"]
    if "audio" in args.save_type:  # create folder for segment
        os.makedirs(os.path.join(args.result_dir, "audio_segment"), exist_ok=True)
    if "spec" in args.save_type:  # list to store audio specs
        audio_specs = []

    for i in df.index:
        if i % 100 == 0:
            print(i)

        if args.seg_type == "word":
            onset = df.audio_onset[i]
            offset = df.audio_offset[i]

        elif args.seg_type == "sentence":
            if df.word[i][-1] in "!?.":  # end of a sentence
                offset = df.audio_offset[i]  # offset
                onset, prev_offset = prev_offset, offset  # reset
                onset_idx = df.audio_onset.ge(onset).idxmax()  # start idx
                label = " ".join(df.loc[onset_idx:i, "word"].tolist())
                words.append(label)
                if offset - onset > 30:  # skip long sentence
                    print(label)
                    continue
            else:  # middle of a sentence
                continue

        elif args.seg_type == "chunk":
            onset, offset, label = get_chunk_data(df, i, "audio_onset", "audio_offset")
            words.append(label)
            assert offset - onset <= 30

        chunk_name = os.path.join(
            args.result_dir, "audio_segment", f"segment_{i:04d}-{df.word[i]}.wav"
        )
        if "audio" in args.save_type:
            # Split and saving wav files
            chunk_data = full_audio[int(onset * fs) : int(offset * fs)]
            wavfile.write(chunk_name, fs, chunk_data)

        if "spec" in args.save_type:
            # Calculating Spectrograms
            assert os.path.isfile(chunk_name), "No audio file"
            audio_specs.append(audio_to_spec(args, chunk_name))

    if "spec" in args.save_type:
        if args.seg_type == "word":
            words = df.word.tolist()  # word level
        # df_index = df.index.tolist()
        # index = np.arange(0, len(df.index))
        assert len(words) == len(audio_specs)
        
        # save spectrogram to pickle
        result = {
            "audio_specs": audio_specs,
            "label": words,
            # "index": index,
            # "df_index": df_index,
        }
        pkl_dir = os.path.join(args.result_dir, "audio_spec")
        save_pickle(result, pkl_dir)

    return None


def main():

    # Read command line arguments
    args = parse_arguments()

    # Setup paths to data and results
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(vars(args))

    # Split audio and save results
    split_audio(args)

    return None


if __name__ == "__main__":
    main()
