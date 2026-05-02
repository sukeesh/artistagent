# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Flow field: every point has an angle from noise; trace particles along it."""
import numpy as np
import matplotlib.pyplot as plt

def value_noise_2d(shape, scale, seed):
    rng = np.random.default_rng(seed)
    coarse = rng.random((shape[0] // scale + 2, shape[1] // scale + 2))
    # bilinear upsample
    ys = np.linspace(0, coarse.shape[0] - 1, shape[0])
    xs = np.linspace(0, coarse.shape[1] - 1, shape[1])
    y0, x0 = ys.astype(int), xs.astype(int)
    y1 = np.minimum(y0 + 1, coarse.shape[0] - 1)
    x1 = np.minimum(x0 + 1, coarse.shape[1] - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = coarse[y0][:, x0] * (1 - fx) + coarse[y0][:, x1] * fx
    b = coarse[y1][:, x0] * (1 - fx) + coarse[y1][:, x1] * fx
    return a * (1 - fy) + b * fy

def fbm_2d(shape, seed, octaves=5):
    out = np.zeros(shape)
    amp, scale = 1.0, 80
    for i in range(octaves):
        out += amp * value_noise_2d(shape, max(scale, 2), seed + i)
        amp *= 0.5
        scale //= 2
    return out

def render(n_particles=1500, steps=200, step_size=1.5, size=800):
    field = fbm_2d((size, size), seed=42, octaves=5)
    angles = (field - field.min()) / (field.max() - field.min()) * 2 * np.pi * 2

    rng = np.random.default_rng(7)
    px = rng.uniform(0, size, n_particles)
    py = rng.uniform(0, size, n_particles)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0b0a1a')
    ax.set_facecolor('#0b0a1a')

    # color each particle by where it started
    colors = plt.cm.magma(rng.uniform(0.3, 0.9, n_particles))

    for _ in range(steps):
        ix = np.clip(px.astype(int), 0, size - 1)
        iy = np.clip(py.astype(int), 0, size - 1)
        a = angles[iy, ix]
        nx = px + np.cos(a) * step_size
        ny = py + np.sin(a) * step_size
        ax.plot(np.stack([px, nx]), np.stack([py, ny]),
                color='white', alpha=0.04, linewidth=0.5)
        # individual colored dots every few steps add sparkle
        px, py = nx, ny

    ax.scatter(px, py, c=colors, s=1.5, alpha=0.6)
    ax.set_xlim(0, size); ax.set_ylim(0, size)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('flow_field.png', dpi=150, bbox_inches='tight',
                facecolor='#0b0a1a')
    plt.show()

if __name__ == '__main__':
    render()
