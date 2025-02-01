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
    x_proj = x / (1 - z)
    y_proj = y / (1 - z)
    return x_proj, y_proj

# Parallel transport on the unit sphere (with unit-length preservation)
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
    
    # Normalize the vector to maintain unit length (this is the key fix)
    transport /= np.linalg.norm(transport)
    
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
solution = solve_ivp(parallel_transport, [0, 2 * np.pi], initial_conditions, args=(theta_0,), t_eval=ϕ, atol=1e-12, rtol=1e-12)

# Compute stereographic projection for each transported vector
proj_vectors = [stereographic_projection(vec[0], vec[1], vec[2]) for vec in solution.y.T]

# Compute the inner products between two vectors after transport
inner_products_sphere = []
inner_products_proj = []

for i in range(len(solution.y[0]) - 1):
    # Calculate the inner product on the sphere
    inner_prod_sphere = np.dot(solution.y[:, i], solution.y[:, i+1])
    inner_products_sphere.append(inner_prod_sphere)
    
    # Stereographic projection of the vectors
    x_proj_1, y_proj_1 = stereographic_projection(solution.y[0, i], solution.y[1, i], solution.y[2, i])
    x_proj_2, y_proj_2 = stereographic_projection(solution.y[0, i+1], solution.y[1, i+1], solution.y[2, i+1])
    
    # Normalize the projected vectors for accurate comparison
    norm_1 = np.sqrt(x_proj_1**2 + y_proj_1**2)
    norm_2 = np.sqrt(x_proj_2**2 + y_proj_2**2)
    x_proj_1 /= norm_1
    y_proj_1 /= norm_1
    x_proj_2 /= norm_2
    y_proj_2 /= norm_2
    
    # Calculate the inner product after stereographic projection
    inner_prod_proj = x_proj_1 * x_proj_2 + y_proj_1 * y_proj_2
    inner_products_proj.append(inner_prod_proj)

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the inner products on the sphere
axs[0].plot(range(len(inner_products_sphere)), inner_products_sphere, color='blue', label='Inner Product on Sphere')
axs[0].set_title('Inner Product on the Sphere')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Inner Product')
axs[0].legend()

# Plot the inner products after stereographic projection
axs[1].plot(range(len(inner_products_proj)), inner_products_proj, color='red', label='Inner Product after Projection')
axs[1].set_title('Inner Product after Stereographic Projection')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Inner Product')
axs[1].legend()

plt.tight_layout()
plt.savefig("inner_product.png")
