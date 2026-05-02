# Drawing nature with pure math

A collection of small Python scripts that render images and scenes from mathematical equations alone — no 3D engine, no textures, no assets.

Every script is self-contained with `uv` inline metadata. Run any of them with:

```bash
uv run <script>.py
```

## Scripts

| File | What it draws | Trick |
| --- | --- | --- |
| `mountains.py` | Layered mountain silhouette | Sums of sine waves at irrational frequency ratios |
| `flow_field.py` | Silky flowing particle art | Particles drift along angles sampled from 2D fBm noise |
| `clifford.py` | Strange attractor | Iterating `sin/cos` equations 2M times, log-density histogram |
| `fern.py` | Barnsley fern | 4 affine maps picked at random (IFS) |
| `domain_warp.py` | Marble / nebula texture | fBm fed into fBm fed into fBm |
| `terrain_3d.py` | Shaded 3D terrain | Heightmap raymarching with Lambert shading |
| `nature_scene.py` | Daytime mountains + lake | Adds Fresnel water, soft shadows, cloud layer |
| `sunset_scene.py` | Sunset over a lake | Low sun, warm sky, glitter path on water |
| `equations_poster.py` | Poster of the equations used | Matplotlib mathtext |

## Core ideas

- **fBm** — sum of noise at doubling frequencies and halving amplitudes
- **Heightmap raymarching** — one ray per pixel, march until it dips below the terrain
- **Schlick's Fresnel** — `F0 + (1 - F0)(1 - cosθ)^5` for water reflectance
- **Soft shadows** — track minimum clearance along the shadow ray
- **Tone mapping + gamma** — `(c/(c+1))^(1/2.2)`

Pure Python + numpy + matplotlib.
