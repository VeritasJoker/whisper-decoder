import argparse
import os


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--sid", nargs="?", type=int, default=None)
    parser.add_argument("--datum", nargs="?", type=str, default=None)
    parser.add_argument("--audio", nargs="?", type=str, default=None)
    parser.add_argument("--save-type", type=str, default=None)

    # args for segmentation
    parser.add_argument("--seg-type", type=str, default=None)

    # args for spectrogram
    parser.add_argument("--sample-rate", nargs="?", type=int, default=16000)
    parser.add_argument("--chunk-len", nargs="?", type=int, default=30)
    parser.add_argument("--hop-len", nargs="?", type=int, default=160)
    parser.add_argument("--n-fft", nargs="?", type=int, default=400)
    parser.add_argument("--n-mel", nargs="?", type=int, default=80)

    args = parser.parse_args()

    assert args.hop_len == 160
    args.n_sample = args.sample_rate * args.chunk_len  # sample num in audio
    assert args.n_sample % args.hop_len == 0
    args.n_frame = args.n_sample // args.hop_len  # frame num in spectrogram

    return args


def setup_environ(args):

    # data directories
    args.data_dir = os.path.join("data", args.project_id)
    args.datum_path = os.path.join(args.data_dir, args.datum)
    args.audio_path = os.path.join(args.data_dir, args.audio)

    # result directories
    args.result_dir = os.path.join("seg-data", args.project_id, args.seg_type)
    os.makedirs(args.result_dir, exist_ok=True)

    return args
