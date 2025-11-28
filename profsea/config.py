"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import pathlib
import yaml
import argparse


# Create argument parser
parser = argparse.ArgumentParser(description="Run script with optional custom path")
parser.add_argument("--path", type=str, help="Supply the directory path (optional)")

# Parse arguments
args = parser.parse_args()


# Use user-supplied path if given, else fallback to script's directory
if args.path:
    path = pathlib.Path(args.path).as_posix()
else:
    path = pathlib.Path(__file__).parents[0].as_posix()


with open(os.path.join(path, "user-settings-emu.yml"), "r") as f:
    settings = yaml.load(f, Loader=yaml.SafeLoader)

print(f"Using path: {path}")
