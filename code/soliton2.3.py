import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Choose initial parameters
b0 = 1.0
# f0 = -1.0

# Small starting point to avoid the singularity at t = 0
eps = 1e-6
t_end = 10.0


def graph(t, y):
    a, ap, b, bp, f, fp = y

    # Replace the tiny interval at t=0 by the exact IC values
    a[0] = 0
    ap[0] = 1
    b[0] = b0
    bp[0] = 0
    f[0] = 0
    fp[0] = 0

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    ax[0].plot(t, a, label=r"$a(t)$")
    ax[0].plot(t, b, label=r"$b(t)$")
    ax[0].set_ylabel("a, b")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, f, label=r"$f(t)$")
    ax[1].set_ylabel("f")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t, ap, label=r"$a'(t)$")
    ax[2].plot(t, bp, label=r"$b'(t)$")
    ax[2].plot(t, fp, label=r"$f'(t)$")
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("derivatives")
    ax[2].legend()
    ax[2].grid()

    plt.tight_layout()
    plt.show()
    return


def ode(t, y):
    a, ap, b, bp, f, fp = y

    # a''
    app = -2 * ap * bp / b + ap * fp + a

    # b''
    bpp = (1 - bp**2) / b - ap * bp / a + bp * fp + b

    # f''
    fpp = app / a + 2 * bpp / b - 1

    return [ap, app, bp, bpp, fp, fpp]

f0 = -10.

for f0 in [-.1, -.5, -1., -2., -5., -10.]:
    # Regular Taylor expansion near t = 0
    a_eps  = eps + (f0 - 1/b0**2) * eps**3 / 6
    ap_eps = 1 + (f0 - 1/b0**2) * eps**2 / 2

    b_eps  = b0 + (b0 + 1/b0) * eps**2 / 4
    bp_eps = (b0 + 1/b0) * eps / 2

    f_eps  = f0 * eps**2 / 2
    fp_eps = f0 * eps

    y0 = [
        a_eps,
        ap_eps,
        b_eps,
        bp_eps,
        f_eps,
        fp_eps,
    ]

    sol = solve_ivp(
        ode,
        (eps, t_end),
        y0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
        max_step=0.01,
    )
    t = np.linspace(0, t_end, 5000)
    y = sol.sol(np.maximum(t, eps))
    print(f"(f0 = {f0}) Final b': ", sol.y[3, -1])
    graph(t, y)