import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the unit sphere
def unit_sphere_mesh():
    u = np.linspace(0, np.pi, 100)  # Latitude
    v = np.linspace(0, 2 * np.pi, 100)  # Longitude
    u, v = np.meshgrid(u, v)
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    return x, y, z

# Stereographic projection
def stereographic_projection(x, y, z):
    # Prevent division by zero for points at the north pole (z = 1)
    z = np.clip(z, -1 + 1e-10, 1 - 1e-10)  # Clipping to avoid division by zero
    x_proj = x / (1 - z)
    y_proj = y / (1 - z)
    return x_proj, y_proj

# Parallel transport on the unit sphere
def parallel_transport(t, vec, theta_0):
    # The tangent vector to the great circle in the phi direction at fixed theta_0
    curve_tangent = np.array([0, np.cos(theta_0), np.sin(theta_0)])  # Tangent along ϕ
    
    # Normalize the tangent vector to make it a unit vector
    curve_tangent /= np.linalg.norm(curve_tangent)
    
    # Ensure the vector to be transported is also a 3D vector
    vec = np.array(vec)
    
    # Cross product for perpendicular component
    cross_product = np.cross(curve_tangent, vec)
    
    # Normalize the cross product (to ensure we stay on the unit sphere)
    cross_product /= np.linalg.norm(cross_product)
    
    # The parallel transport condition (transport the vector while staying parallel to the surface)
    transport = np.cross(cross_product, curve_tangent)
    
    return transport

# Create the unit sphere mesh
x, y, z = unit_sphere_mesh()

# Define initial conditions for the vector n(θ_0, ϕ = 0)
theta_0 = np.pi / 5  # Initial theta near the north pole
alpha = 1.0
beta = np.sqrt(1 - alpha**2)  # Ensuring the normalization condition is met

# Define the initial vector at (θ_0, ϕ = 0)
n_θ0 = np.array([alpha, beta * np.sin(theta_0), 0])  # αe_θ + β sin(θ_0) e_ϕ

# Define the closed loop (parametrization from ϕ = 0 to ϕ = 2π)
num_points = 100
ϕ = np.linspace(0, 2 * np.pi, num_points)  # Parametrize the curve

gc_x = np.sin(theta_0) * np.cos(ϕ)
gc_y = np.sin(theta_0) * np.sin(ϕ)
gc_z = np.cos(theta_0) * np.ones(num_points)

# Solve for parallel transport along the loop using numerical integration
initial_conditions = n_θ0
solution = solve_ivp(parallel_transport, [0, 2 * np.pi], initial_conditions, args=(theta_0,), t_eval=ϕ)

# Compute stereographic projection for each transported vector
proj_vectors = [stereographic_projection(vec[0], vec[1], vec[2]) for vec in solution.y.T]

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the unit sphere and the parallel transported vectors
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x, y, z, color='lightblue', alpha=0.5)
ax.quiver(gc_x, gc_y, gc_z, solution.y[0], solution.y[1], solution.y[2], length=0.05, normalize=True)

ax.set_title('Parallel Transport on the Unit Sphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot the stereographic projection of the parallel transport vectors
axs[1].plot(proj_vectors[0], proj_vectors[1], color='red', label='Parallel Transported Vectors')
axs[1].set_title('Stereographic Projection of Parallel Transport')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].legend()

plt.tight_layout()
plt.savefig("parallel_transport_trajectory.png")
