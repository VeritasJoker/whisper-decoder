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
    parser.add_argument("--save-type", type=str, default=None)

    # args for segmentation
    parser.add_argument("--elecs", nargs="*", type=int)
    parser.add_argument("--elec-type", type=str, default="all")
    parser.add_argument("--seg-type", type=str, default=None)
    parser.add_argument("--onset-shift", nargs="?", type=int, default=0)
    parser.add_argument("--window-size", nargs="?", type=int, default=625)

    # args for spectrogram
    parser.add_argument("--hop-len", nargs="?", type=int, default=10)
    parser.add_argument("--n-fft", nargs="?", type=int, default=400)
    parser.add_argument("--n-mel", nargs="?", type=int, default=80)

    args = parser.parse_args()

    args.fs = 512
    # bin_ms = 62.5
    # bin_fs = int(bin_ms / 1000 * fs)
    args.shift_fs = int(args.onset_shift / 1000 * args.fs)
    args.window_fs = int(args.window_size / 1000 * args.fs)
    args.half_window = int(args.window_fs // 2)
    # n_bins = window_ms // bin_ms
    # n_bins = window_fs // bin_fs

    args.n_frame = 3000
    args.n_sample = args.n_frame * args.hop_len  # frame num in spectrogram

    return args


def setup_environ(args):

    # data directories
    args.data_dir = os.path.join("data", args.project_id)
    args.datum_path = os.path.join(args.data_dir, args.datum)
    args.ecog_path = f"/projects/HASSON/247/data/podcast/NY{args.sid}_111_Part1_conversation1/preprocessed"

    # result directories
    args.result_dir = os.path.join("seg-data", args.project_id, args.seg_type)
    os.makedirs(args.result_dir, exist_ok=True)

    return args
