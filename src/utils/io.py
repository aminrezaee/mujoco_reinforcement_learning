import json
from os import listdir
from typing import Dict
import shutil
import yaml
import os


def read_config(yaml_file):
    with open(yaml_file, "r") as file:
        yaml_data = file.read()
        documents = yaml.safe_load_all(yaml_data)
    return documents


def read_json_config(json_file):
    with open(json_file, "r") as file:
        json_file = json.load(file)
    return json_file


def save_config(config, path):
    with open(path, "w") as file:
        json.dump(config, file)
    return


def find_experiment_name(experiment_id: int, experiments_directory: str) -> str:
    directories = listdir(experiments_directory)
    for directory in directories:
        directory_id = directory.split("_")[0]
        if int(directory_id) == experiment_id:
            return directory.split(f"{experiment_id}_")[1]


def __get_summarized_word(phrase: str):
    summarized_word = ""
    words = phrase.split("_")
    for word in words:
        summarized_word = summarized_word + word[0:2]
    return summarized_word


def _get_dict_name(dictionary: Dict):
    name = ""
    for key in dictionary.keys():

        name = name + f"{__get_summarized_word(key)}_"
        if isinstance(dictionary[key], Dict):
            name = name + f"_{_get_dict_name(dictionary[key])}_"
        else:
            name = name + f"{int(dictionary[key])}"

    return name


def add_episode_to_best_results(experiment_path: str, current_episode: int):
    shutil.copytree(f"{experiment_path}/networks/{current_episode}",
                    f"{experiment_path}/networks/best_results/{current_episode}")
    if os.path.exists(f"{experiment_path}/visualizations/{current_episode}"):
        shutil.copytree(f"{experiment_path}/visualizations/{current_episode}",
                        f"{experiment_path}/visualizations/best_results/{current_episode}")


def remove_epoch_results(experiment_path: str, removing_epoch: int):
    if os.path.exists(f"{experiment_path}/networks/{removing_epoch}"):
        shutil.rmtree(f"{experiment_path}/networks/{removing_epoch}")
    # if os.path.exists(f"{run.experiment_path}/visualizations/{removing_epoch}"):
    #     shutil.rmtree(f"{run.experiment_path}/visualizations/{removing_epoch}")
    if os.path.exists(f"{experiment_path}/debugs/{removing_epoch}"):
        shutil.rmtree(f"{experiment_path}/debugs/{removing_epoch}")
