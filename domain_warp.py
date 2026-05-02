# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Domain warping: feed noise into noise → marble / nebula / lava textures."""
import numpy as np
import matplotlib.pyplot as plt

def hash2d(x, y, seed):
    # cheap deterministic hash → values in [0,1)
    h = np.sin(x * 127.1 + y * 311.7 + seed * 17.3) * 43758.5453
    return h - np.floor(h)

def value_noise(x, y, seed):
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    # smoothstep
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    a = hash2d(xi,     yi,     seed)
    b = hash2d(xi + 1, yi,     seed)
    c = hash2d(xi,     yi + 1, seed)
    d = hash2d(xi + 1, yi + 1, seed)
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v

def fbm(x, y, seed, octaves=5):
    out = np.zeros_like(x)
    amp, freq = 1.0, 1.0
    for i in range(octaves):
        out += amp * value_noise(x * freq, y * freq, seed + i)
        amp *= 0.5; freq *= 2.0
    return out

def render(size=600):
    lin = np.linspace(0, 4, size)
    x, y = np.meshgrid(lin, lin)

    # Inigo Quilez's classic recipe: warp the domain twice
    q1 = fbm(x, y, seed=1)
    q2 = fbm(x + 5.2, y + 1.3, seed=2)
    r1 = fbm(x + 4 * q1 + 1.7, y + 4 * q2 + 9.2, seed=3)
    r2 = fbm(x + 4 * q1 + 8.3, y + 4 * q2 + 2.8, seed=4)
    final = fbm(x + 4 * r1, y + 4 * r2, seed=5)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(final, cmap='twilight_shifted', origin='lower')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('domain_warp.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    render()
