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
    # Projection from the north pole (0, 0, 1)
    return x / (1 - z), y / (1 - z)

# Function to compute the tangent vectors at points on the sphere (finite differences)
def compute_tangent_vectors_finite_diff(x, y, z):
    # Ensure that the inputs are 1D arrays
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    # Add zero padding to maintain the same length as the input
    dx = np.append(dx, 0)
    dy = np.append(dy, 0)
    dz = np.append(dz, 0)
    return dx, dy, dz

# Function to compute the angle between two vectors
def compute_angle_safe(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    # Avoid division by zero and ensure the cosine value is within the range [-1, 1]
    if norm1 == 0 or norm2 == 0:
        return np.nan
    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta) * 180 / np.pi

# Redefine stereographic projection for tangent vectors
def stereographic_tangent_projection(tangent_x, tangent_y, tangent_z):
    proj_tangent_x = tangent_x / (1 - tangent_z)
    proj_tangent_y = tangent_y / (1 - tangent_z)
    return proj_tangent_x, proj_tangent_y

# Create the unit sphere mesh
x, y, z, u, v = unit_sphere_mesh()

# Define two intersecting curves: latitude (constant latitude) and longitude (constant longitude)
latitude_curve = np.linspace(0, np.pi, 100)
longitude_curve = np.linspace(0, 2 * np.pi, 100)

# Parametric equations for the curves on the unit sphere
lat_x = np.sin(latitude_curve) * np.cos(np.pi / 4)
lat_y = np.sin(latitude_curve) * np.sin(np.pi / 4)
lat_z = np.cos(latitude_curve)

# Properly define lon_z as a 1D array (not a scalar)
lon_x = np.sin(np.pi / 4) * np.cos(longitude_curve)
lon_y = np.sin(np.pi / 4) * np.sin(longitude_curve)
lon_z = np.cos(longitude_curve)  # This is now correctly defined as a 1D array!

# Compute the tangent vectors for the curves on the unit sphere using finite differences
lat_tangent_x, lat_tangent_y, lat_tangent_z = compute_tangent_vectors_finite_diff(lat_x, lat_y, lat_z)
lon_tangent_x, lon_tangent_y, lon_tangent_z = compute_tangent_vectors_finite_diff(lon_x, lon_y, lon_z)

# Compute the angle between the tangent vectors at the intersection points on the sphere
valid_index = 10  # Use a middle index that is safe
angle_on_sphere = compute_angle_safe(
    np.array([lat_tangent_x[valid_index], lat_tangent_y[valid_index], lat_tangent_z[valid_index]]),
    np.array([lon_tangent_x[valid_index], lon_tangent_y[valid_index], lon_tangent_z[valid_index]])
)

# Project the tangent vectors
lat_proj_tangent_x, lat_proj_tangent_y = stereographic_tangent_projection(lat_tangent_x, lat_tangent_y, lat_tangent_z)
lon_proj_tangent_x, lon_proj_tangent_y = stereographic_tangent_projection(lon_tangent_x, lon_tangent_y, lon_tangent_z)

# Compute the angle between the projected tangent vectors in the x-y plane
angle_after_projection = compute_angle_safe(
    np.array([lat_proj_tangent_x[valid_index], lat_proj_tangent_y[valid_index]]), 
    np.array([lon_proj_tangent_x[valid_index], lon_proj_tangent_y[valid_index]])
)

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the unit sphere and the curves
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x, y, z, color='lightblue', alpha=0.5)
ax.plot(lat_x, lat_y, lat_z, color='red', label='Latitude Curve')
ax.plot(lon_x, lon_y, lon_z, color='green', label='Longitude Curve')

# Add annotation for angle on the unit sphere
ax.text(lat_x[valid_index], lat_y[valid_index], lat_z[valid_index], f"{angle_on_sphere:.2f}°", color='black')

ax.set_title('Curves on the Unit Sphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Plot the stereographic projection (P')
lat_proj_x, lat_proj_y = stereographic_projection(lat_x, lat_y, lat_z)
lon_proj_x, lon_proj_y = stereographic_projection(lon_x, lon_y, lon_z)

axs[1].plot(lat_proj_x, lat_proj_y, color='red', label='Latitude Curve (P\'s)')
axs[1].plot(lon_proj_x, lon_proj_y, color='green', label='Longitude Curve (P\'s)')

# Add annotation for angle after projection
axs[1].text(lat_proj_x[valid_index], lat_proj_y[valid_index], f"{angle_after_projection:.2f}°", color='black')

axs[1].set_title('Stereographic Projection on the XY Plane')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].legend()

plt.tight_layout()
plt.savefig("parallel_transport_a.png")

print(f"Angle on the unit sphere: {angle_on_sphere} degrees")
print(f"Angle after stereographic projection: {angle_after_projection} degrees")
