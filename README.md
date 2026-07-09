# Duffing Oscillator Residual Learning Demo

This implements Bai-style model-error correction for the Duffing oscillator.

The Duffing equation used here is:

```text
x_dot = v
v_dot = -delta * v - alpha * x - beta * x^3 + gamma * cos(omega * t)
```

The demo compares three trajectories:

- `truth`: RK4 integration with the trusted Duffing parameters.
- `baseline`: one Euler step with biased Duffing parameters.
- `corrected`: baseline prediction plus a learned residual correction.

The residual model is:

```text
StandardScaler -> RBFSampler -> Ridge
```

It is trained on one-step labels:

```text
truth_next_state - baseline_next_state
```

Latest verified run:

- One-step training samples: 9100
- Mean baseline error over rollout: 0.5049
- Mean corrected error over rollout: 0.0089
- Final baseline error: 1.3667
- Final corrected error: 0.0172

This is a cleaner example than the pendulum if the main goal is explaining the
machine-learning residual idea, because there is no angle wrapping.
