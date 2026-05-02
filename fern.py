# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Barnsley fern: 4 affine maps picked at random → a fern."""
import numpy as np
import matplotlib.pyplot as plt

# (a, b, c, d, e, f, weight) — each row is an affine transform
TRANSFORMS = np.array([
    [ 0.00,  0.00,  0.00,  0.16, 0.00, 0.00, 0.01],
    [ 0.85,  0.04, -0.04,  0.85, 0.00, 1.60, 0.85],
    [ 0.20, -0.26,  0.23,  0.22, 0.00, 1.60, 0.07],
    [-0.15,  0.28,  0.26,  0.24, 0.00, 0.44, 0.07],
])

def fern(n=500_000, seed=1):
    rng = np.random.default_rng(seed)
    weights = TRANSFORMS[:, 6]
    choices = rng.choice(4, size=n, p=weights)
    xs = np.empty(n); ys = np.empty(n)
    x, y = 0.0, 0.0
    for i in range(n):
        a, b, c, d, e, f, _ = TRANSFORMS[choices[i]]
        x, y = a * x + b * y + e, c * x + d * y + f
        xs[i] = x; ys[i] = y
    return xs, ys

def render():
    xs, ys = fern()
    fig, ax = plt.subplots(figsize=(6, 10), facecolor='#0a0f08')
    ax.scatter(xs, ys, s=0.05, c='#5fc26a', alpha=0.6)
    ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout()
    plt.savefig('fern.png', dpi=150, bbox_inches='tight', facecolor='#0a0f08')
    plt.show()

if __name__ == '__main__':
    render()
