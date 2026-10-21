
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import linregress

# Initial conditions
f0 = 0.0
alpha = 0.0  # a'(0)
beta = 0.0   # b'(0)
gamma = 0.0  # f'(0)

# Perturbations to the initial slopes
delta_alpha = 1e-3
delta_beta = 1e-3
delta_gamma = 1e-3

T = 20.0
N = 3000
t_eval = np.linspace(0.0, T, N)

# Stop integration before a or b reaches zero.
EPS = 1e-8


def rhs(t, y):
    a, b, p, q, f, r = y

    # Avoid evaluating the singular vector field.
    if a <= EPS or b <= EPS:
        raise ValueError("The solution approached a singular boundary.")

    # a'' and b''
    pp = -2.0 * p * q / b + p * r + a
    qq = (1.0 - q**2) / b - p * q / a + q * r + b

    # f''
    rr = pp / a + 2.0 * qq / b - 1.0

    return np.array([p, q, pp, qq, r, rr])


def event_a(t, y):
    return y[0] - EPS


def event_b(t, y):
    return y[1] - EPS


for event in (event_a, event_b):
    event.terminal = True
    event.direction = -1


def integrate(y0):
    return solve_ivp(
        rhs,
        (0.0, T),
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        events=(event_a, event_b),
        max_step=0.02,
    )


# Reference solution
y0 = np.array([1.0, 1.0, alpha, beta, f0, gamma])
sol = integrate(y0)

# Perturbed solution
y0_pert = y0 + np.array([
    0.0, 0.0,
    delta_alpha, delta_beta,
    0.0, delta_gamma
])
pert = integrate(y0_pert)

if not sol.success:
    print("Reference integration:", sol.message)
if not pert.success:
    print("Perturbed integration:", pert.message)

# Compare only times available in both integrations.
n = min(sol.t.size, pert.t.size)
t = sol.t[:n]

Y = sol.y[:, :n]
Yp = pert.y[:, :n]
difference = Yp - Y

# Plot the reference solution
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
labels = [r"$a$", r"$b$", r"$a'$", r"$b'$", r"$f$", r"$f'$"]

for i, ax in enumerate(axes.flat):
    ax.plot(t, Y[i], label="Reference")
    ax.plot(t, Yp[i], "--", label="Perturbed")
    ax.set_ylabel(labels[i])
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1, 0].set_xlabel("t")
axes[-1, 1].set_xlabel("t")
fig.tight_layout()
plt.show()

# Norm of the difference between the two states
# State ordering: (a, b, a', b', f, f')
err = np.linalg.norm(difference, axis=0)

plt.figure(figsize=(8, 5))
plt.semilogy(t, np.maximum(err, 1e-30))
plt.xlabel("t")
plt.ylabel(r"$\|\Delta X(t)\|_2$")
plt.title("Growth or decay of the perturbation")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# Approximate finite-time growth rate
if len(t) > 1 and err[0] > 0:
    rate = np.full_like(err, np.nan)
    valid = (err > 0) & (t > 0)
    rate[valid] = np.log(err[valid] / err[0]) / t[valid]

    plt.figure(figsize=(8, 5))
    plt.plot(t[valid], rate[valid])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("t")
    plt.ylabel("Finite-time growth rate")
    plt.title("Average perturbation growth rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Choose the time interval to fit.
# Adjust these after inspecting your graph.
t_start = 1.0
t_end = 10.0

mask = (
    (t >= t_start)
    & (t <= t_end)
    & np.isfinite(err)
    & (err > 0)
)

tf = t[mask]
Ef = err[mask]

if len(tf) < 3:
    raise ValueError("Not enough valid points in the fitting interval.")

# Fit log(E) = intercept + lambda * t
logE = np.log(Ef)
fit = linregress(tf, logE)

lam = fit.slope
C = np.exp(fit.intercept)
r_squared = fit.rvalue**2

print(f"Estimated exponential rate lambda = {lam:.6g}")
print(f"Estimated prefactor C = {C:.6g}")
print(f"R-squared = {r_squared:.6f}")

# Plot the data and exponential fit
t_fit = np.linspace(tf[0], tf[-1], 300)
E_fit = C * np.exp(lam * t_fit)

plt.figure(figsize=(8, 5))
plt.semilogy(t, np.maximum(err, 1e-300),
             label="Numerical perturbation")
plt.semilogy(t_fit, E_fit, "--",
             label=fr"Fit: $Ce^{{\lambda t}}$, "
                   fr"$\lambda={lam:.4g}$")
plt.axvline(t_start, color="gray", linestyle=":")
plt.axvline(t_end, color="gray", linestyle=":")
plt.xlabel("t")
plt.ylabel(r"$E(t)=\|\Delta X(t)\|_2$")
plt.title("Exponential fit of perturbation growth")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()