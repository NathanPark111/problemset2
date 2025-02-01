import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the unit sphere
def unit_sphere_mesh():
    u = np.linspace(0, np.pi, 100)  # Latitude
    v = np.linspace(0, 2 * np.pi, 100)  # Longitude
    u, v = np.meshgrid(u, v)
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    return x, y, z, u, v

# Stereographic projection
def stereographic_projection(x, y, z):
    # Prevent division by zero for points at the north pole (z = 1)
    z = np.clip(z, -1 + 1e-10, 1 - 1e-10)  # Clipping to avoid division by zero
    x_proj = x / (1 - z)
    y_proj = y / (1 - z)
    return x_proj, y_proj

# Function to create great circles
def great_circle(center, axis, num_points=100):
    # Parametrize the great circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_x = center[0] * np.cos(theta) + axis[0] * np.sin(theta)
    circle_y = center[1] * np.cos(theta) + axis[1] * np.sin(theta)
    circle_z = center[2] * np.cos(theta) + axis[2] * np.sin(theta)
    return circle_x, circle_y, circle_z

# Create the unit sphere mesh
x, y, z, u, v = unit_sphere_mesh()

# Define a new great circle on the equator (x-y plane)
center_1 = [0, 0, 1]  # North pole at (0, 0, 1)
axis_1 = [1, 0, 0]    # Axis along x (equator)
gc_x, gc_y, gc_z = great_circle(center_1, axis_1)

# Define a great circle on the x-z plane
center_3 = [0, 1, 0]  # Center at (0, 1, 0)
axis_3 = [1, 0, 0]    # Axis along x
gc_3_x, gc_3_y, gc_3_z = great_circle(center_3, axis_3)

# Apply stereographic projection to the great circles
gc_x_proj, gc_y_proj = stereographic_projection(gc_x, gc_y, gc_z)
gc_3_x_proj, gc_3_y_proj = stereographic_projection(gc_3_x, gc_3_y, gc_3_z)

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the unit sphere and the great circles (x-y and x-z planes only)
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x, y, z, color='lightblue', alpha=0.5)
ax.plot(gc_x, gc_y, gc_z, color='red', label='Great Circle 1 (x-y plane)')
ax.plot(gc_3_x, gc_3_y, gc_3_z, color='blue', label='Great Circle 3 (x-z plane)')

ax.set_title('Great Circles on the Unit Sphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Plot the stereographic projection of the great circles (x-y and x-z planes only)
axs[1].plot(gc_x_proj, gc_y_proj, color='red', label='Great Circle 1 (x-y plane)')
axs[1].plot(gc_3_x_proj, gc_3_y_proj, color='blue', label='Great Circle 3 (x-z plane)')

axs[1].set_title('Stereographic Projection of Great Circles')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].legend()

plt.tight_layout()
plt.savefig("greatcircle.png")
