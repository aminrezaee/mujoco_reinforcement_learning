import os
from argparse import ArgumentParser
from typing import Any, Dict

import torch
from torch.utils.tensorboard.writer import SummaryWriter


def add_historgrams(input: Any, writer: SummaryWriter, full_key: str = ""):
    if isinstance(input, Dict):
        for key in input.keys():
            add_historgrams(input[key], writer, full_key=full_key + key)
    else:
        writer.add_histogram(full_key, input, 0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-tp", '--tensorboard_path', type=str)
    args = parser.parse_args()
    os.makedirs(args.tensorboard_path, exist_ok=True)
    state_dict = torch.load(args.path)
    writer = SummaryWriter(log_dir=args.tensorboard_path)
    add_historgrams(state_dict, writer, "")
    writer.close()
