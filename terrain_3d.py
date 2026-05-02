# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""3D terrain via heightmap raymarching. Pure numpy, no 3D engine.

Per pixel: cast a ray from camera, march until it dips below the heightmap,
compute the normal from the heightmap gradient, shade with sun + fog + sky.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- noise ----------
def hash2(x, y):
    h = np.sin(x * 127.1 + y * 311.7) * 43758.5453
    return h - np.floor(h)

def noise2(x, y):
    xi, yi = np.floor(x), np.floor(y)
    xf, yf = x - xi, y - yi
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    a = hash2(xi, yi);     b = hash2(xi + 1, yi)
    c = hash2(xi, yi + 1); d = hash2(xi + 1, yi + 1)
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v

def terrain(x, z):
    """fbm with ridged low octaves for sharp peaks, smooth high octaves for detail."""
    h = np.zeros_like(x)
    amp, freq = 1.0, 0.05
    for i in range(7):
        n = noise2(x * freq, z * freq)
        if i < 3:
            n = 1 - np.abs(2 * n - 1)  # ridge
            n = n * n
        h += amp * n
        amp *= 0.5
        freq *= 2.0
    return h * 8.0 - 1.5

# ---------- camera ----------
def make_rays(W, H, cam, target, fov_deg=55):
    forward = target - cam; forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 1, 0])); right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    aspect = W / H
    fov_scale = np.tan(np.radians(fov_deg) / 2)
    px = ((np.arange(W) + 0.5) / W * 2 - 1) * aspect * fov_scale
    py = (1 - (np.arange(H) + 0.5) / H * 2) * fov_scale
    px, py = np.meshgrid(px, py)

    rd = (forward[:, None, None]
          + right[:, None, None] * px
          + up[:, None, None] * py)
    rd /= np.linalg.norm(rd, axis=0)
    ro = np.broadcast_to(cam[:, None, None], (3, H, W)).copy()
    return ro, rd

# ---------- raymarch ----------
def march(ro, rd, max_t=120, steps=140):
    H, W = ro.shape[1], ro.shape[2]
    t = np.full((H, W), 0.5)
    p = ro + rd * t
    h_prev = p[1] - terrain(p[0], p[2])
    t_prev = t.copy()
    hit = np.zeros((H, W), dtype=bool)
    t_hit = np.full((H, W), max_t)

    for _ in range(steps):
        p = ro + rd * t
        h_now = p[1] - terrain(p[0], p[2])
        crossed = (~hit) & (h_now < 0) & (h_prev > 0)
        frac = h_prev / (h_prev - h_now + 1e-8)
        t_cross = t_prev + frac * (t - t_prev)
        t_hit = np.where(crossed, t_cross, t_hit)
        hit |= crossed

        t_prev = t.copy()
        h_prev = h_now
        # adaptive step: proportional to clearance, with floor that grows with distance
        step = np.maximum(0.15 + 0.01 * t, h_now * 0.4)
        t = np.where(hit, t, t + step)
        if (t > max_t).all() or hit.all():
            break
    t_hit = np.where(hit, t_hit, max_t)
    return hit, t_hit

# ---------- shading ----------
def normal_at(x, z, eps=0.08):
    dx = terrain(x + eps, z) - terrain(x - eps, z)
    dz = terrain(x, z + eps) - terrain(x, z - eps)
    n = np.stack([-dx, np.full_like(dx, 2 * eps), -dz])
    n /= np.linalg.norm(n, axis=0) + 1e-9
    return n

def sky(rd, sun):
    horizon = np.array([0.95, 0.78, 0.62])
    zenith  = np.array([0.30, 0.48, 0.78])
    t = np.clip(rd[1] * 0.5 + 0.5, 0, 1)
    col = horizon[:, None, None] * (1 - t)[None] + zenith[:, None, None] * t[None]
    sun_dot = np.clip(rd[0] * sun[0] + rd[1] * sun[1] + rd[2] * sun[2], 0, 1)
    col = col + np.array([1.0, 0.85, 0.65])[:, None, None] * (sun_dot ** 64) * 1.2
    col = col + np.array([1.0, 0.7, 0.45])[:, None, None]  * (sun_dot ** 4)  * 0.15
    return col

def render(W=480, H=270):
    cam = np.array([0.0, 4.0, -10.0])
    target = np.array([0.0, 2.5, 0.0])
    sun = np.array([0.6, 0.5, 0.3]); sun /= np.linalg.norm(sun)

    ro, rd = make_rays(W, H, cam, target, fov_deg=55)
    hit, t_hit = march(ro, rd)

    p = ro + rd * t_hit
    n = normal_at(p[0], p[2])

    # diffuse + ambient sky
    diff = np.clip(n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2], 0, 1)
    ambient = np.clip(n[1], 0, 1) * 0.4

    # albedo: blend grass/rock by slope, snow up high
    height_n = np.clip((p[1] + 1.5) / 9.0, 0, 1)
    slope = 1 - n[1]
    grass = np.array([0.32, 0.42, 0.22])
    rock  = np.array([0.42, 0.36, 0.30])
    snow  = np.array([0.92, 0.94, 0.97])

    s = np.clip(slope * 1.4, 0, 1)
    albedo = grass[:, None, None] * (1 - s) + rock[:, None, None] * s
    snow_mask = (height_n > 0.55) & (slope < 0.45)
    snow_mask = snow_mask.astype(float) * np.clip((height_n - 0.55) / 0.2, 0, 1)
    albedo = albedo * (1 - snow_mask) + snow[:, None, None] * snow_mask

    sun_color = np.array([1.0, 0.92, 0.78])[:, None, None]
    sky_amb   = np.array([0.45, 0.55, 0.75])[:, None, None]
    color = albedo * (sun_color * diff + sky_amb * ambient)

    # atmospheric fog
    fog = 1 - np.exp(-t_hit * 0.025)
    sky_col = sky(rd, sun)
    color = color * (1 - fog) + sky_col * fog
    color = np.where(hit[None], color, sky_col)

    # tone map + gamma
    color = color / (color + 1.0)
    color = np.clip(color, 0, 1) ** (1 / 2.2)

    img = np.transpose(color, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(12, 12 * H / W))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('terrain_3d.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    render()
