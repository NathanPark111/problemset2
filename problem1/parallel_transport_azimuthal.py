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

def spherical_basis_to_cartesian(theta, phi):
    e_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_r, e_theta, e_phi

def parallel_transport_equation_phi(phi, n, theta_0):
    n_theta, n_phi = n
    dn_theta = np.cos(theta_0) * n_phi
    dn_phi = -np.cos(theta_0) * n_theta
    return [dn_theta, dn_phi]

def parallel_transport_vector_phi(alpha, beta, theta_0, phi_0, phi_f, steps=100):
    phi_vals = np.linspace(phi_0, phi_f, steps)
    sol = solve_ivp(parallel_transport_equation_phi, [phi_0, phi_f], [alpha, beta], args=(theta_0,), t_eval=phi_vals)
    transported_vectors = []
    path_points = []
    
    for i, phi in enumerate(phi_vals):
        x, y, z = spherical_to_cartesian(1, theta_0, phi)
        _, e_theta, e_phi = spherical_basis_to_cartesian(theta_0, phi)
        n_vec = sol.y[0, i] * e_theta + sol.y[1, i] * e_phi
        transported_vectors.append((x, y, z, n_vec))
        path_points.append((x, y, z))
    
    return transported_vectors, path_points

def plot_parallel_transport_phi(theta_0, phi_f=2*np.pi, steps=100):
    alpha = np.cos(theta_0)
    beta = np.sin(theta_0)
    transported_vectors, path_points = parallel_transport_vector_phi(alpha, beta, theta_0, 0, phi_f, steps)
    
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
    arrow_scale = 0.5  # Scale down arrow length
    for i in range(0, len(transported_vectors), arrow_step):
        x, y, z, vec = transported_vectors[i]
        ax.quiver(x, y, z, vec[0] * arrow_scale, vec[1] * arrow_scale, vec[2] * arrow_scale, color='b')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig("parallel_transport_2.png")

# Example: Transport a vector along φ from ϕ=0 to ϕ=2π at θ=π/5
plot_parallel_transport_phi(theta_0=np.pi/5, phi_f=2*np.pi)
