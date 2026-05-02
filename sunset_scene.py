# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Sunset variant: low sun on the horizon, long shadows, warm sky, glitter on lake."""
import numpy as np
import matplotlib.pyplot as plt

# =============== noise ===============
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

def fbm2(x, y, octaves=5, freq=0.02, amp=1.0):
    out = np.zeros_like(x)
    for _ in range(octaves):
        out += amp * noise2(x * freq, y * freq)
        amp *= 0.5; freq *= 2.0
    return out

def terrain(x, z):
    h = np.zeros_like(x)
    amp, freq = 1.0, 0.04
    for i in range(7):
        n = noise2(x * freq, z * freq)
        if i < 3:
            n = 1 - np.abs(2 * n - 1); n = n * n
        h += amp * n
        amp *= 0.5; freq *= 2.0
    return h * 9.0 - 2.5

# =============== camera ===============
def make_rays(W, H, cam, target, fov_deg=55):
    forward = target - cam; forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0, 1, 0])); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    aspect = W / H; fs = np.tan(np.radians(fov_deg) / 2)
    px = ((np.arange(W) + 0.5) / W * 2 - 1) * aspect * fs
    py = (1 - (np.arange(H) + 0.5) / H * 2) * fs
    px, py = np.meshgrid(px, py)
    rd = (forward[:, None, None] + right[:, None, None] * px + up[:, None, None] * py)
    rd /= np.linalg.norm(rd, axis=0)
    ro = np.broadcast_to(cam[:, None, None], (3, H, W)).copy()
    return ro, rd

# =============== marching ===============
def march_terrain(ro, rd, max_t=140, steps=160):
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
        t_hit = np.where(crossed, t_prev + frac * (t - t_prev), t_hit)
        hit |= crossed
        t_prev = t.copy(); h_prev = h_now
        step = np.maximum(0.2 + 0.012 * t, h_now * 0.4)
        t = np.where(hit, t, t + step)
        if (t > max_t).all() or hit.all():
            break
    return hit, np.where(hit, t_hit, max_t)

def soft_shadow(p, sun, steps=28, max_t=40):
    H, W = p.shape[1], p.shape[2]
    res = np.ones((H, W))
    t = np.full((H, W), 0.4)
    sun_b = sun[:, None, None]
    for _ in range(steps):
        q = p + sun_b * t
        h = q[1] - terrain(q[0], q[2])
        res = np.minimum(res, 4.0 * h / t)
        t = t + np.maximum(0.4, h * 0.5)
        if (t > max_t).all():
            break
    return np.clip(res, 0.08, 1.0)

def normal_at(x, z, eps=0.08):
    dx = terrain(x + eps, z) - terrain(x - eps, z)
    dz = terrain(x, z + eps) - terrain(x, z - eps)
    n = np.stack([-dx, np.full_like(dx, 2 * eps), -dz])
    n /= np.linalg.norm(n, axis=0) + 1e-9
    return n

def water_normal(x, z):
    nx = np.zeros_like(x); nz = np.zeros_like(x)
    for k, (fx, fz, ph) in enumerate([(0.8, 0.3, 0.0), (0.5, 0.9, 1.4),
                                       (1.3, 0.7, 2.7), (0.4, 1.1, 4.1),
                                       (1.7, 0.2, 5.6)]):
        a = 0.045 / (k + 1)
        c = np.cos(fx * x + fz * z + ph)
        nx += -a * fx * c
        nz += -a * fz * c
    n = np.stack([nx, np.ones_like(nx), nz])
    n /= np.linalg.norm(n, axis=0)
    return n

# =============== sky & clouds ===============
def sky(rd, sun):
    """Three-band sunset sky: deep zenith, pink mid, fiery horizon, with sun disk."""
    t = np.clip(rd[1] * 0.5 + 0.5, 0, 1)
    horizon = np.array([1.05, 0.55, 0.28])  # orange
    mid     = np.array([0.95, 0.45, 0.42])  # pink
    zenith  = np.array([0.18, 0.18, 0.42])  # deep blue/purple

    # smooth blend across two stops
    band = np.clip((t - 0.0) / 0.35, 0, 1)
    col_low  = horizon[:, None, None] * (1 - band)[None] + mid[:, None, None] * band[None]
    band2 = np.clip((t - 0.35) / 0.55, 0, 1)
    col = col_low * (1 - band2)[None] + zenith[:, None, None] * band2[None]

    sun_dot = np.clip(rd[0] * sun[0] + rd[1] * sun[1] + rd[2] * sun[2], 0, 1)
    # broad warm bloom
    col = col + np.array([1.2, 0.55, 0.25])[:, None, None] * (sun_dot ** 4) * 0.55
    # tighter halo
    col = col + np.array([1.4, 0.85, 0.55])[:, None, None] * (sun_dot ** 32) * 0.9
    # sun disk
    col = col + np.array([1.6, 1.2, 0.85])[:, None, None] * (sun_dot ** 800) * 4.0
    return col

def clouds(rd, ro, sun):
    cloud_y = 22.0
    valid = rd[1] > 0.02
    t = (cloud_y - ro[1]) / np.where(valid, rd[1], 1.0)
    px = ro[0] + rd[0] * t
    pz = ro[2] + rd[2] * t
    n = fbm2(px, pz, octaves=5, freq=0.022)
    density = np.clip((n - 0.50) * 2.2, 0, 1)
    density = np.where(valid, density, 0.0)
    # backlit by low sun: edges glow, bodies stay dark
    sun_dot = np.clip(rd[0] * sun[0] + rd[1] * sun[1] + rd[2] * sun[2], 0, 1)
    rim = sun_dot ** 6
    base = np.array([0.30, 0.22, 0.32])[:, None, None]   # dark underside
    lit  = np.array([1.30, 0.75, 0.45])[:, None, None]   # warm rim
    color = base * (1 - rim)[None] + lit * rim[None]
    return density, color

# =============== render ===============
def render(W=440, H=250):
    cam     = np.array([0.0, 3.2, -13.0])
    target  = np.array([0.5, 1.8,   0.0])
    # low sun, mostly aligned with camera-forward so glitter falls on the lake toward us
    sun     = np.array([0.25, 0.13, 0.95]); sun /= np.linalg.norm(sun)
    water_y = 0.4

    ro, rd = make_rays(W, H, cam, target, fov_deg=55)
    print("marching terrain...")
    hit_t, t_t = march_terrain(ro, rd)

    denom = np.where(np.abs(rd[1]) < 1e-4, -1e-4, rd[1])
    t_w = (water_y - ro[1]) / denom
    water_hit = (t_w > 0.5) & (rd[1] < 0) & ((~hit_t) | (t_w < t_t))

    is_water   = water_hit
    is_terrain = hit_t & ~water_hit
    is_sky     = ~(is_terrain | is_water)
    t_hit = np.where(is_water, t_w, np.where(is_terrain, t_t, 200.0))
    p = ro + rd * t_hit

    # ----- terrain -----
    print("shading terrain (long shadows)...")
    n_t = normal_at(p[0], p[2])
    diff = np.clip(np.einsum('ihw,i->hw', n_t, sun), 0, 1)
    ambient = np.clip(n_t[1], 0, 1) * 0.35
    shadow = soft_shadow(p, sun)
    diff = diff * shadow

    height_above = p[1] - water_y
    slope = 1 - n_t[1]
    grass = np.array([0.22, 0.28, 0.16])[:, None, None]
    rock  = np.array([0.38, 0.30, 0.26])[:, None, None]
    snow  = np.array([0.95, 0.85, 0.85])[:, None, None]   # pink-tinted
    sand  = np.array([0.65, 0.50, 0.38])[:, None, None]

    s = np.clip(slope * 1.5, 0, 1)
    albedo = grass * (1 - s) + rock * s
    sand_mask = np.clip(1 - height_above / 0.7, 0, 1) ** 2
    albedo = albedo * (1 - sand_mask) + sand * sand_mask
    snow_t = (np.clip((height_above - 4.5) / 1.5, 0, 1)
              * np.clip(1 - slope * 2.0, 0, 1))
    albedo = albedo * (1 - snow_t) + snow * snow_t

    # warm low sun, cool dim ambient from purple zenith
    sun_col = np.array([1.40, 0.70, 0.40])[:, None, None]
    sky_amb = np.array([0.30, 0.28, 0.45])[:, None, None]
    terrain_color = albedo * (sun_col * diff + sky_amb * ambient)

    # ----- water -----
    print("shading water (glitter path)...")
    n_w = water_normal(p[0], p[2])
    rdotN = np.einsum('ihw,ihw->hw', rd, n_w)
    refl = rd - 2 * rdotN[None] * n_w
    sky_refl = sky(refl, sun)
    fres = 0.02 + 0.98 * (1 - np.clip(-rdotN, 0, 1)) ** 5
    h_below = np.clip(water_y - terrain(p[0], p[2]), 0, 5)
    depth = np.clip(h_below * 0.4, 0, 1)
    deep    = np.array([0.04, 0.06, 0.14])[:, None, None]
    shallow = np.array([0.18, 0.22, 0.32])[:, None, None]
    water_base = shallow * (1 - depth) + deep * depth
    # broad sun glitter — wide power so it forms a streak across rippled normals
    glint = np.clip(np.einsum('ihw,i->hw', refl, sun), 0, 1)
    spec = (glint ** 24) * 1.4 + (glint ** 200) * 3.5
    water_color = water_base * (1 - fres)[None] + sky_refl * fres[None]
    water_color = water_color + np.array([1.5, 0.95, 0.65])[:, None, None] * spec[None]

    # ----- sky + clouds -----
    print("sky & clouds...")
    sky_col = sky(rd, sun)
    cloud_d, cloud_c = clouds(rd, ro, sun)
    sky_col = sky_col * (1 - cloud_d)[None] + cloud_c * cloud_d[None]

    # ----- combine -----
    color = np.where(is_terrain[None], terrain_color, sky_col)
    color = np.where(is_water[None], water_color, color)

    # warm-tinted fog (haze picks up sunset color)
    fog = 1 - np.exp(-np.clip(t_hit, 0, 200) * 0.028)
    fog = np.where(is_sky, 0, fog)
    fog_col = sky_col * 0.7 + np.array([0.6, 0.35, 0.25])[:, None, None] * 0.3
    color = color * (1 - fog)[None] + fog_col * fog[None]

    # tone map + gamma
    color = color / (color + 1.0)
    color = np.clip(color, 0, 1) ** (1 / 2.2)

    img = np.transpose(color, (1, 2, 0))
    fig, ax = plt.subplots(figsize=(13, 13 * H / W))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('sunset_scene.png', dpi=160, bbox_inches='tight')
    plt.show()
    print("saved sunset_scene.png")

if __name__ == '__main__':
    render()
