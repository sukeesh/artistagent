# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Clifford strange attractor: x,y = sin(a*y) + c*cos(a*x), sin(b*x) + d*cos(b*y)."""
import numpy as np
import matplotlib.pyplot as plt

def clifford(a, b, c, d, n=2_000_000):
    xs = np.empty(n); ys = np.empty(n)
    x, y = 0.1, 0.0
    for i in range(n):
        x, y = np.sin(a * y) + c * np.cos(a * x), np.sin(b * x) + d * np.cos(b * y)
        xs[i] = x; ys[i] = y
    return xs, ys

def render():
    # try also: (-1.7, 1.8, -1.9, -0.4), (-1.4, 1.6, 1.0, 0.7)
    a, b, c, d = -1.8, -2.0, -0.5, -0.9
    xs, ys = clifford(a, b, c, d)

    # 2D histogram → density image (much faster than scatter)
    h, xe, ye = np.histogram2d(xs, ys, bins=1000)
    h = np.log1p(h)  # log-scale density for visual range

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.imshow(h.T, origin='lower', cmap='inferno',
              extent=[xe[0], xe[-1], ye[0], ye[-1]])
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('clifford.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.show()

if __name__ == '__main__':
    render()
