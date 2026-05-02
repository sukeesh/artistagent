# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
import numpy as np
import matplotlib.pyplot as plt

def generate_heightmap(resolution):
    """Generates a base heightmap using a simplified Perlin-like noise."""
    def perlin_noise(x, y, octaves=4, persistence=0.5, amplitude=1.0):
        noise = 0.0
        max_amplitude = 0.0
        frequency = 1.0
        for _ in range(octaves):
            noise += amplitude * np.sin(x * frequency + y * frequency)
            max_amplitude += amplitude
            frequency *= 2
            amplitude *= persistence
        return noise / max_amplitude

    heightmap = np.zeros((resolution, resolution))
    for x in range(resolution):
        for y in range(resolution):
            heightmap[x, y] = perlin_noise(x / resolution, y / resolution, octaves=4)
    return heightmap

def domain_warp(heightmap, octaves=3):
    """Applies domain warping to the heightmap."""
    warped_heightmap = np.zeros_like(heightmap, dtype=np.float32)
    for i in range(octaves):
        frequency = 2**i
        warped_heightmap += np.sin(heightmap * frequency) / (frequency + 0.1) #Avoid division by zero
    return heightmap + warped_heightmap * 0.1

def raymarch(heightmap, camera_pos, sun_angle, resolution=512, steps=1024):
    """Raymarches the scene to generate the image."""

    heightmap = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))  # Normalize
    heightmap = heightmap * 5  # Scale for more pronounced terrain

    image = np.zeros((resolution, resolution, 3))
    for x in range(resolution):
        for y in range(resolution):
            ray_direction = np.array([x - resolution * 0.5, y - resolution * 0.5, -1])  # Camera at (0,0,-1)
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            ray_origin = np.array([0, 0, 0])

            for step in range(steps):
                ray_pos = ray_origin + ray_direction * step
                height = heightmap[int(ray_pos[0]), int(ray_pos[1])]
                ray_pos[2] = height

                if np.linalg.norm(ray_pos) > 5:  # Clip distance
                    break

            if np.linalg.norm(ray_pos) < 5:
                # Lighting calculation
                normal = np.cross(np.array([1, 0, 0]), ray_direction)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                else:
                    normal = np.array([0,0,1])

                light_direction = np.array([1, 1, -1])
                light_direction = light_direction / np.linalg.norm(light_direction)

                diffuse = max(0, np.dot(normal, light_direction))
                specular = max(0, np.dot(normal, -ray_direction))**50

                # Colors based on height and lighting
                height_factor = (ray_pos[2] - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))
                color = 0.2 + 0.8 * height_factor
                image[x, y] = np.array([color, color * 0.8, color * 0.5]) * diffuse + specular * 0.2

    return image

# Constants
resolution = 512
heightmap_resolution = 256
domain_warp_octaves = 3
raymarching_steps = 256
sun_angle = 20

# Generate and warp heightmap
heightmap = generate_heightmap(heightmap_resolution)
warped_heightmap = domain_warp(heightmap, octaves=domain_warp_octaves)

# Raymarch and generate image
image = raymarch(warped_heightmap, camera_pos=np.array([0, 0, -2]), sun_angle=sun_angle, resolution=resolution, steps=raymarching_steps)

# Display the image
plt.imshow(image)
plt.axis('off')
plt.savefig('output.png')
plt.show()