import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_cylindrical(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi, z

def cylindrical_to_cartesian(rho, phi, z):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

def spherical_to_cylindrical(r, theta, phi):
    rho = r * np.sin(theta)
    z = r * np.cos(theta)
    return rho, phi, z

def cylindrical_to_spherical(rho, phi, z):
    r = np.sqrt(rho**2 + z**2)
    theta = np.arctan2(rho, z)
    return r, theta, phi

def spherical_basis_to_cartesian(theta, phi):
    e_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_r, e_theta, e_phi

def cylindrical_basis_to_cartesian(phi):
    e_rho = np.array([np.cos(phi), np.sin(phi), 0])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    e_z = np.array([0, 0, 1])
    return e_rho, e_phi, e_z

def local_orthonormal_basis_on_sphere(theta, phi):
    e_r, e_theta, e_phi = spherical_basis_to_cartesian(theta, phi)
    return {
        'Position Vector': e_r,
        'Theta Basis': e_theta,
        'Phi Basis': e_phi
    }

def plot_sphere_with_basis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.3)
    
    # Sample point on the sphere
    theta, phi = np.pi / 4, np.pi / 4
    r = 1
    px, py, pz = spherical_to_cartesian(r, theta, phi)
    e_r, e_theta, e_phi = spherical_basis_to_cartesian(theta, phi)
    
    # Plot position vector
    ax.quiver(0, 0, 0, px, py, pz, color='r', label='Position Vector')
    # Plot spherical basis vectors
    ax.quiver(px, py, pz, *e_r, color='b', label='e_r')
    ax.quiver(px, py, pz, *e_theta, color='g', label='e_theta')
    ax.quiver(px, py, pz, *e_phi, color='m', label='e_phi')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig("unit_sphere.png")

# Call the function to plot
# plot_sphere_with_basis()

def generate_local_coordinate_system(f, x_range, y_range, dx=0.01, dy=0.01):
    x = np.arange(x_range[0], x_range[1], dx)
    y = np.arange(y_range[0], y_range[1], dy)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    dZdx, dZdy = np.gradient(Z, dx, dy, axis=(1, 0))
    normal = np.dstack((-dZdx, -dZdy, np.ones_like(Z)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    
    tangent_x = np.dstack((np.ones_like(Z), np.zeros_like(Z), dZdx))
    tangent_x /= np.linalg.norm(tangent_x, axis=2, keepdims=True)
    
    tangent_y = np.dstack((np.zeros_like(Z), np.ones_like(Z), dZdy))
    tangent_y /= np.linalg.norm(tangent_y, axis=2, keepdims=True)
    
    return X, Y, Z, normal, tangent_x, tangent_y

def plot_surface_with_basis(f, x_range, y_range, dx=0.5, dy=0.5):
    X, Y, Z, normal, tangent_x, tangent_y = generate_local_coordinate_system(f, x_range, y_range, dx, dy)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='c', alpha=0.3)
    
    for i in range(0, X.shape[0], 5):
        for j in range(0, X.shape[1], 5):
            ax.quiver(X[i, j], Y[i, j], Z[i, j], *normal[i, j], color='r')
            ax.quiver(X[i, j], Y[i, j], Z[i, j], *tangent_x[i, j], color='g')
            ax.quiver(X[i, j], Y[i, j], Z[i, j], *tangent_y[i, j], color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("surface_plot.png")

def example_surface(x, y):
    return np.sin(x) * np.cos(y)

# Call function to visualize
# plot_surface_with_basis(example_surface, (-2, 2), (-2, 2))

def parallel_transport_equation(theta, n, beta):
    n_theta, n_phi = n
    dn_theta = -beta * np.cos(theta) * n_phi
    dn_phi = beta * np.cos(theta) * n_theta
    return [dn_theta, dn_phi]

def parallel_transport_vector(alpha, beta, theta_0, theta_f, steps=100):
    theta_vals = np.linspace(theta_0, theta_f, steps)
    sol = solve_ivp(parallel_transport_equation, [theta_0, theta_f], [alpha, beta * np.sin(theta_0)], args=(beta,), t_eval=theta_vals)
    transported_vectors = []
    path_points = []
    
    for i, theta in enumerate(theta_vals):
        x, y, z = spherical_to_cartesian(1, theta, 0)
        _, e_theta, e_phi = spherical_basis_to_cartesian(theta, 0)
        n_vec = sol.y[0, i] * e_theta + sol.y[1, i] * e_phi
        
        # Project onto the tangent plane to ensure parallel transport follows unit-speed motion
        n_vec -= np.dot(n_vec, np.array([np.sin(theta), 0, np.cos(theta)])) * np.array([np.sin(theta), 0, np.cos(theta)])
        transported_vectors.append((x, y, z, n_vec))
        path_points.append((x, y, z))
    
    return transported_vectors, path_points

def plot_parallel_transport(theta_0, theta_f, steps=100):
    # Ensure norm constraint is satisfied
    norm_n = 1  # Assume unit-norm vector
    alpha = np.cos(theta_0)
    beta = np.sqrt((norm_n**2 - alpha**2) / np.sin(theta_0)**2) if np.sin(theta_0) != 0 else 0
    
    transported_vectors, path_points = parallel_transport_vector(alpha, beta, theta_0, theta_f, steps)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(X, Y, Z, color='c', alpha=0.3)
    
    path_points = np.array(path_points)
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'r-', label='Transport Path')
    
    arrow_step = max(len(transported_vectors) // 10, 1)  # Show around 10 arrows
    arrow_scale = 0.5  # Arrow length
    for i in range(0, len(transported_vectors), arrow_step):
        x, y, z, vec = transported_vectors[i]
        ax.quiver(x, y, z, vec[0] * arrow_scale, vec[1] * arrow_scale, vec[2] * arrow_scale, color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig("parallel_transport_1.png")

# Example: Transport a vector from θ=π/5 to θ=π/2
plot_parallel_transport(theta_0=np.pi/5, theta_f=np.pi/2)