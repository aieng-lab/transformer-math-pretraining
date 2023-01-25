from comet_ml import Experiment
from datetime import datetime


def create_experiment(project_name, exp_name):
    experiment = Experiment(
        api_key="zmkYcmj6YSa7BXwnvj6oEdBHA",
        project_name=project_name,
        workspace="nasisnaefellsnes",
        #auto_metric_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_log_co2=True,
        auto_histogram_tensorboard_logging=True,
        log_env_details=True,
        log_env_cpu=True,
        log_env_gpu=True,
        log_env_host=True
    )

    experiment.set_name(exp_name)

    return experiment


def get_exp_name(text=None):
    current_date = datetime.now()
    dt_string = current_date.strftime("%Y-%m-%d %H:%M:%S")
    if text is None:
        name = f"{dt_string} :: Experiment"
    else:
        name = f"{dt_string} :: Experiment"
    return name
