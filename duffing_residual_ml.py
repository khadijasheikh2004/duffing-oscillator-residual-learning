"""
Residual learning demo for the Duffing oscillator.

This is the same Bai-style model-error correction idea as the pendulum demo:

    corrected_next_state = baseline_next_state + ML_predicted_baseline_error

The machine learning model does not learn the whole Duffing oscillator from
scratch. It learns the residual error left by a cheaper, biased physics model.

Run:
    python duffing_residual_ml.py

Outputs:
    duffing_residual_results.png
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DuffingParams:
    """Parameters for a forced damped Duffing oscillator.

    The equation is:

        x_dot = v
        v_dot = -delta * v - alpha * x - beta * x^3
                + gamma * cos(omega * t)

    With alpha < 0 and beta > 0, the potential is a double well. Under periodic
    forcing, this system is a classic example of nonlinear and chaotic motion.
    """

    delta: float
    alpha: float
    beta: float
    gamma: float
    omega: float


# High-fidelity reference dynamics. In a real problem, this could be a trusted
# simulator, measured data, or a more expensive physical model.
TRUTH = DuffingParams(
    delta=0.20,
    alpha=-1.00,
    beta=1.00,
    gamma=0.32,
    omega=1.20,
)

# Cheap baseline model. It is deliberately biased: damping, stiffness, nonlinear
# stiffness, and forcing are all slightly wrong.
BASELINE = DuffingParams(
    delta=0.17,
    alpha=-0.92,
    beta=0.88,
    gamma=0.29,
    omega=1.20,
)


def duffing_rhs(t: float, y: np.ndarray, params: DuffingParams) -> np.ndarray:
    """Evaluate the right-hand side of the Duffing ODE."""

    x, v = y
    dx = v
    dv = (
        -params.delta * v
        - params.alpha * x
        - params.beta * x**3
        + params.gamma * np.cos(params.omega * t)
    )
    return np.array([dx, dv])


def rk4_step(t: float, y: np.ndarray, dt: float, params: DuffingParams) -> np.ndarray:
    """Advance one step using fourth-order Runge-Kutta integration."""

    k1 = duffing_rhs(t, y, params)
    k2 = duffing_rhs(t + 0.5 * dt, y + 0.5 * dt * k1, params)
    k3 = duffing_rhs(t + 0.5 * dt, y + 0.5 * dt * k2, params)
    k4 = duffing_rhs(t + dt, y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def baseline_step(t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Cheap one-step predictor using biased Duffing parameters.

    The baseline uses a single Euler step. That intentionally mixes two kinds of
    error: wrong physics parameters and a lower-order numerical integrator.
    """

    return y + dt * duffing_rhs(t, y, BASELINE)


def truth_step(t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """High-fidelity target step used to generate supervised labels."""

    return rk4_step(t, y, dt, TRUTH)


def make_features(t: float, y: np.ndarray, baseline_next: np.ndarray) -> np.ndarray:
    """Build ML inputs for predicting the baseline model's one-step error.

    Features include the current state, nonlinear state terms, forcing phase,
    and the baseline model's own next-state prediction. That last part is the
    key Bai-style idea: use the simple model as an input to the error model.
    """

    x, v = y
    base_x, base_v = baseline_next
    phase = TRUTH.omega * t
    return np.array(
        [
            x, # current position
            v, # current velocity
            x**2, # nonlinear position term
            x**3, # duffing nonlinearity
            x * v, # interaction term
            np.sin(phase), # forcing sine term
            np.cos(phase), # forcing cosine term
            base_x, # baseline position prediction
            base_v, # baseline velocity prediction
            base_x**3, # baseline nonlinear term
        ]
    )


def build_one_step_dataset(
    rng: np.random.Generator,
    n_trajectories: int,
    steps_per_trajectory: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one-step residual-learning examples.

    Each label is:

        true_next_state - baseline_next_state

    The model then learns how to correct the baseline model locally.
    """

    x_rows = [] # stores features 
    y_rows = [] # stores labels, labels are the residuals between truth and baseline
    for _ in range(n_trajectories):
        t = rng.uniform(0.0, 40.0)
        state = np.array([rng.uniform(-1.8, 1.8), rng.uniform(-1.2, 1.2)])
        for _ in range(steps_per_trajectory):
            base_next = baseline_step(t, state, dt)
            true_next = truth_step(t, state, dt)
            x_rows.append(make_features(t, state, base_next)) # stores features
            y_rows.append(true_next - base_next) # machine learning target (residual = truth - baseline)
            state = true_next
            t += dt
    return np.vstack(x_rows), np.vstack(y_rows)


def fit_residual_model(train_x: np.ndarray, train_y: np.ndarray):
    """Fit a fast nonlinear residual regressor.

    `RBFSampler + Ridge` approximates an RBF-kernel regression model. It is close
    in spirit to using an RBF SVR for residual prediction, but faster and simpler
    for a compact demonstration.
    """

    return make_pipeline(
        StandardScaler(),
        RBFSampler(gamma=0.7, n_components=450, random_state=7),
        Ridge(alpha=1e-3),
    ).fit(train_x, train_y)


def corrected_step(t: float, state: np.ndarray, dt: float, model) -> np.ndarray:
    """Predict one step with baseline physics plus learned residual correction."""

    base_next = baseline_step(t, state, dt)
    correction = model.predict(make_features(t, state, base_next)[None, :])[0]
    return base_next + correction


def rollout(
    initial_state: np.ndarray,
    t0: float,
    dt: float,
    n_steps: int,
    model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out truth, baseline-only, and corrected trajectories side by side."""

    times = t0 + dt * np.arange(n_steps + 1)
    truth = np.zeros((n_steps + 1, 2))
    baseline = np.zeros_like(truth)
    corrected = np.zeros_like(truth)
    truth[0] = initial_state
    baseline[0] = initial_state
    corrected[0] = initial_state

    for i in range(n_steps):
        t = times[i]
        truth[i + 1] = truth_step(t, truth[i], dt)
        baseline[i + 1] = baseline_step(t, baseline[i], dt)
        corrected[i + 1] = corrected_step(t, corrected[i], dt, model)
    return times, truth, baseline, corrected


def state_error(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Euclidean error in position/velocity state space."""

    # represents "how far away" the prediction is from the truth in the combined position–velocity state space.
    return np.linalg.norm(predicted - truth, axis=1) # sqrt((x_predicted - x_truth)^2 + (v_predicted - v_truth)^2)
                                                    

def main() -> None:
    """Train, evaluate, and plot the Duffing residual-learning demo."""

    rng = np.random.default_rng(22)
    dt = 0.03

    train_x, train_y = build_one_step_dataset(
        rng,
        n_trajectories=35,
        steps_per_trajectory=260,
        dt=dt,
    )
    model = fit_residual_model(train_x, train_y)

    # Duffing can be chaotic, so this is a short-term trajectory forecast test.
    # Longer tests should use rolling-window errors or attractor statistics.
    initial_state = np.array([0.35, 0.10])
    times, truth, baseline, corrected = rollout(
        initial_state=initial_state,
        t0=4.0,
        dt=dt,
        n_steps=500,
        model=model,
    ) # run the three simulations (truth, baseline, corrected) from the same initial state.

    baseline_error = state_error(baseline, truth) # compute how far the baseline simulation is from the truth.
    corrected_error = state_error(corrected, truth) # compute how far the corrected simulation is from the truth.
    horizon = times - times[0]

    print("One-step training samples:", len(train_x))
    print("Mean baseline error over rollout:  ", baseline_error.mean())
    print("Mean corrected error over rollout: ", corrected_error.mean())
    print("Final baseline error:              ", baseline_error[-1])
    print("Final corrected error:             ", corrected_error[-1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # compares the true position, the baseline prediction, and the corrected prediction over time.
    ax = axes[0, 0]
    ax.plot(horizon, truth[:, 0], label="truth", color="#1b1b1f", linewidth=2)
    ax.plot(horizon, baseline[:, 0], label="baseline", color="#d95f02", alpha=0.9)
    ax.plot(horizon, corrected[:, 0], label="baseline + learned error", color="#1b9e77")
    ax.set_title("Position rollout")
    ax.set_xlabel("seconds")
    ax.set_ylabel("x")
    ax.legend()

    # shows how prediction error changes over time (on a logarithmic scale).
    ax = axes[0, 1]
    ax.semilogy(horizon, baseline_error + 1e-9, label="baseline", color="#d95f02")
    ax.semilogy(horizon, corrected_error + 1e-9, label="corrected", color="#1b9e77")
    ax.set_title("State error")
    ax.set_xlabel("seconds")
    ax.set_ylabel("sqrt(x_error^2 + v_error^2)")
    ax.legend()

    # plots position versus velocity, showing the shape of each trajectory in phase space.
    ax = axes[1, 0]
    ax.plot(truth[:, 0], truth[:, 1], label="truth", color="#1b1b1f", linewidth=2)
    ax.plot(baseline[:, 0], baseline[:, 1], label="baseline", color="#d95f02", alpha=0.8)
    ax.plot(corrected[:, 0], corrected[:, 1], label="corrected", color="#1b9e77", alpha=0.9)
    ax.set_title("Phase portrait")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.legend()

    # compares the true residuals with the residuals predicted by the ML model.
    ax = axes[1, 1]
    sample = slice(None, None, 40)
    predicted_residual = model.predict(train_x[sample])
    ax.scatter(train_y[sample, 0], predicted_residual[:, 0], s=10, alpha=0.35)
    lim = np.max(np.abs(train_y[sample, 0])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="#444444", linewidth=1)
    ax.set_title("Learned one-step position residual")
    ax.set_xlabel("true residual")
    ax.set_ylabel("predicted residual")

    output_path = Path(__file__).with_name("duffing_residual_results.png")
    fig.savefig(output_path, dpi=160)
    print("Saved plot:", output_path)


if __name__ == "__main__":
    main()