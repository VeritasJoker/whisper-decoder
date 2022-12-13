import argparse
import os
import json


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--sid", nargs="?", type=int, default=None)
    parser.add_argument("--seg-type", type=str, default="word")
    parser.add_argument("--model-size", type=str, default="tiny")
    parser.add_argument("--elec-type", type=str, default="")
    parser.add_argument("--ecog-type", type=str, default="all")
    parser.add_argument("--saving-dir", type=str, default=None)
    parser.add_argument("--data-split", type=float, default=0.2)
    parser.add_argument("--eval-file", type=str, default=None)
    parser.add_argument("--eval-model", type=str, default="")

    args = parser.parse_args()

    args.data_dir = os.path.join("seg-data", args.project_id, args.seg_type)
    args.datafile = os.path.join(
        args.data_dir, f"{args.sid}_ecog_{args.elec_type}_spec.pkl"
    )

    return args


def write_model_config(dictionary):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    config_file = os.path.join("models", f"{dictionary['saving_dir']}_config.json")
    with open(config_file, "w") as outfile:
        outfile.write(json_object)
