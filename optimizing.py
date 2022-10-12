# %%
import numpy as np
import cv2
import optuna
from good_rules import CallableRuleset, basic_convolution, fast_inv_gaussian_activation


def run_for(n, field, rules):
    for _ in range(n):
        field = rules(field)
    return field


def build_kernel(a1, a2, a3):
    return np.array([
        [a1, a2, a1],
        [a2, a3, a2],
        [a1, a2, a1]
    ])


def objective(trial: optuna.Trial):
    a1 = trial.suggest_float('corner', -1, 1)
    a2 = trial.suggest_float('side', -1, 1)
    a3 = trial.suggest_float('center', -1, 1)
    field = np.random.binomial(1, p=0.5, size=(800, 800)).astype('float32')
    kernel = build_kernel(a1, a2, a3)
    rules = CallableRuleset(
        kernel,
        basic_convolution,
        fast_inv_gaussian_activation,
        lambda x: x
    )
    t0 = run_for(300, field, rules)

    res = []
    for _ in range(300):
        t1 = run_for(1, t0, rules)
        diff = cv2.calcOpticalFlowFarneback(
            cv2.bilateralFilter(t0, 9, 150, 150),
            cv2.bilateralFilter(t1, 9, 150, 150),
            None, pyr_scale=0.5, levels=3, winsize=5, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
        res.append(cv2.cartToPolar(diff[..., 0], diff[..., 1])[0])

    return np.mean(res)


study = optuna.study.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
# %%
study.trials_dataframe()
# %%
