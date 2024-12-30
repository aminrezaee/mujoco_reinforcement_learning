import json
from os import listdir
from typing import Dict

import yaml


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
