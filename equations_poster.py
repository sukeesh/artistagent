# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Render a poster of the key equations behind nature_scene.py and sunset_scene.py.
Uses matplotlib mathtext (no LaTeX install required).
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'

SECTIONS = [
    ("Fractal Brownian Motion (terrain & clouds)",
     r"$H(x,z) \,=\, \sum_{i=0}^{N-1} \, \frac{1}{2^{i}} \; \mathrm{noise}\left(2^{i} f \cdot x,\; 2^{i} f \cdot z\right)$"),

    ("Ridged octaves for sharp peaks",
     r"$n_{ridge} \,=\, \left(1 - |\,2n - 1\,|\right)^{2}$"),

    ("Heightmap raymarching",
     r"$\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}, \quad \mathrm{find\ smallest\ } t : r_y(t) < H(r_x, r_z)$"),

    ("Surface normal from gradient",
     r"$\mathbf{n} \,\propto\, \left(-\partial_x H,\; 1,\; -\partial_z H\right)$"),

    ("Reflection (water)",
     r"$\mathbf{r} \,=\, \mathbf{d} - 2(\mathbf{d}\cdot\mathbf{n})\,\mathbf{n}$"),

    ("Schlick's Fresnel approximation",
     r"$F(\theta) \,=\, F_{0} + (1 - F_{0})\,(1 - \cos\theta)^{5}, \qquad F_{0} \approx 0.02$"),

    ("Lambertian diffuse + soft shadow",
     r"$L_{diff} \,=\, \max(0,\,\mathbf{n}\cdot\mathbf{s}) \cdot \min_{t>0}\left(\frac{k\,h(t)}{t}\right)$"),

    ("Sun disk and glow on the sky",
     r"$\mathrm{sun}(\mathbf{d}) \,=\, \max(0,\,\mathbf{d}\cdot\mathbf{s})^{\,k}, \qquad k \in \{4,\,32,\,800\}$"),

    ("Water surface normal (sum of sine waves)",
     r"$\mathbf{n}_{w} \,\propto\, \left(-\sum_{i} a_{i} f_{x,i}\cos\phi_{i},\; 1,\; -\sum_{i} a_{i} f_{z,i}\cos\phi_{i}\right)$"),

    ("Atmospheric fog (Beer-Lambert)",
     r"$\mathrm{fog}(t) \,=\, 1 - e^{-\sigma t}$"),

    ("Tone map and gamma",
     r"$c_{out} \,=\, \left(\frac{c}{c + 1}\right)^{1/2.2}$"),
]

def render():
    fig = plt.figure(figsize=(11, 14), facecolor='#0c0a18')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0c0a18')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # title
    ax.text(0.5, 0.965, "Drawing nature, in equations",
            ha='center', va='top', color='#ffd9a8',
            fontsize=26, fontweight='bold', family='serif')
    ax.text(0.5, 0.928, "the math behind nature_scene.py and sunset_scene.py",
            ha='center', va='top', color='#c0a8d0',
            fontsize=13, style='italic', family='serif')

    # equation list
    n = len(SECTIONS)
    top, bottom = 0.88, 0.04
    step = (top - bottom) / n
    for i, (title, eq) in enumerate(SECTIONS):
        y = top - i * step
        ax.text(0.06, y, title, ha='left', va='top',
                color='#f0a87a', fontsize=12.5, family='serif',
                fontweight='bold')
        ax.text(0.10, y - 0.028, eq, ha='left', va='top',
                color='#f4f1ea', fontsize=15, family='serif')

    # footer
    ax.text(0.5, 0.012, "pure python + numpy  ·  no engine, no textures",
            ha='center', va='bottom', color='#7a6a8a',
            fontsize=10, style='italic', family='serif')

    plt.savefig('equations_poster.png', dpi=180,
                bbox_inches='tight', facecolor='#0c0a18')
    plt.show()
    print("saved equations_poster.png")

if __name__ == '__main__':
    render()
