import os


class Logger:
    experiment_path = ""
    REWARD_TYPE = 0
    ACTION_TYPE = 1
    TRAINING_TYPE = 2
    TESTING_TYPE = 3

    @staticmethod
    def get_log_type_str(log_type: int):
        log_types = {0: 'rewards_', 1: 'actions_', 2: 'trainings_', 3: 'testings_'}
        return log_types[log_type]

    @staticmethod
    def get_file_name(historical: bool, validation: bool, log_type: int):
        file_name = ""
        log_type_str = Logger.get_log_type_str(log_type)
        file_name = file_name + log_type_str
        file_name = file_name + 'historical_' if historical else 'multi_agent_environment'
        if validation:
            file_name = file_name + 'validation'
        return f"{file_name}.txt"

    @staticmethod
    def log(message: str, episode: int, log_type: int, path=None, historical: bool = False,
            validation: bool = False, print_message=False):
        mode = "a+"
        if print_message:
            print(message)
        if path is not None:
            Logger.experiment_path = f"{path}/debugs"
            os.makedirs(Logger.experiment_path, exist_ok=True)
            mode = "w"
        if len(Logger.experiment_path) == 0 and path is None:
            raise ValueError("you need to first set the logger path!")
        debug_path = f"{Logger.experiment_path}/{episode}"
        os.makedirs(debug_path, exist_ok=True)
        file_name = Logger.get_file_name(historical, validation, log_type)
        with open(f"{debug_path}/{file_name}", mode) as debug:
            debug.write(f"{message}\n")
            debug.flush()
            debug.close()
