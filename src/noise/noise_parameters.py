
from src.noise.noise_models import depolarizing_thermal_noise_model

SHERBROOKE_PARAMS = dict(
    t1=0.00027841972084544125,
    t2=0.0001700127424244177,
    time_1q=5.6888888888888887e-08,
    time_2q=5.333333333333332e-07,
    err_1q=0.0002362269001245515,
    err_2q=0.00778809624800933,
)


def get_sherbrooke_noise_model():
    return depolarizing_thermal_noise_model(**SHERBROOKE_PARAMS)