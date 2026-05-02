# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
import numpy as np
import matplotlib.pyplot as plt

def ridge(x, seed, octaves=6, base_amp=1.0, base_freq=0.6):
    rng = np.random.default_rng(seed)
    y = np.zeros_like(x)
    amp, freq = base_amp, base_freq
    for _ in range(octaves):
        phase = rng.uniform(0, 2 * np.pi)
        # irrational-ish frequency multiplier keeps layers from aligning
        y += amp * np.sin(freq * x + phase)
        amp *= 0.5
        freq *= 2.17
    return y

def render(layers=6, width=12, height=6):
    x = np.linspace(0, 30, 2000)
    fig, ax = plt.subplots(figsize=(width, height))

    # sky gradient
    sky = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(sky, extent=[x.min(), x.max(), -layers, layers + 1],
              aspect='auto', cmap='RdPu', alpha=0.6, origin='lower', zorder=0)

    for i in range(layers):
        # back layers: lower amplitude, higher base, lighter color
        depth = i / (layers - 1)
        y = ridge(x, seed=i * 7 + 1, octaves=6,
                 base_amp=0.6 + 0.8 * depth,
                 base_freq=0.3 + 0.4 * (1 - depth))
        y += (layers - i) * 0.9  # stack vertically, far layers higher
        shade = 0.15 + 0.7 * (1 - depth)  # closer = darker
        color = (shade * 0.25, shade * 0.2, shade * 0.35)
        ax.fill_between(x, y, -layers, color=color, zorder=i + 1)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-layers + 0.5, layers + 1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('mountains.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    render()
