from comet_ml import Experiment
from datetime import datetime


def create_experiment(project_name, exp_name, api_key):
    if api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace="math-ir-pretraining",
            # auto_metric_logging=True,
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


def get_exp_name(text=None, run_number=None, one_by_one=True, deberta=False, steps=None):
    current_date = datetime.now()
    dt_string = current_date.strftime("%Y-%m-%d %H:%M:%S")
    if text is None:
        name = f"{dt_string} :: Experiment"
    else:
        name = f"{dt_string} :: Experiment"
    if run_number:
        name = f"{name}_{run_number}"

    if one_by_one:
        name = name + '_consecutive'
    else:
        name = name + '_mixed'

    if deberta:
        name = name + '_deberta'
    else:
        name = name + '_bert'

    if steps:
        name = name + '_%dk' % (steps // 1000)

    return name


def get_init_params():
    return {
        "workspace": "math-ir-pretraining",
        "auto_log_co2": True,
        "log_env_details": True,
        "log_env_cpu": True,
        "log_env_gpu": True,
        "log_env_host": True
    }
