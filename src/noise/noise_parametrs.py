
from src.noise.noise_models import depolarizing_thermal_noise_model

def get_sherbrooke_noise_model():
    return depolarizing_thermal_noise_model(
        t1=289.55e-6,
        t2=186.01e-6,
        time_1q=42.67e-9,
        time_2q=539.90e-9,
        depol_1q=0.00042,
        depol_2q=0.07200
    )
