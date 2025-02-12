from argparse import ArgumentParser
import shutil
import os

parser = ArgumentParser()
parser.add_argument("-p", "--path")
args = parser.parse_args()
experiment_path: str = args.path
visualizations_path = f"{experiment_path}/visualizations"
saved_visualizations = [name for name in os.listdir(visualizations_path) if "best" not in name]
for saved_name in saved_visualizations:
    if int(saved_name) % 100 != 0:
        shutil.rmtree(f"{visualizations_path}/{saved_name}")
