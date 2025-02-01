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
    final_vector = sol.y[:, -1]
    initial_vector = np.array([alpha, beta])
    inner_product = np.dot(initial_vector, final_vector)
    return inner_product

def plot_holonomy_strength(theta_values, phi_f=2*np.pi, steps=100):
    holonomy_values = []
    
    for theta_0 in theta_values:
        alpha = np.cos(theta_0)  # Ensure general vector
        beta = np.sin(theta_0)
        holonomy = parallel_transport_vector_phi(alpha, beta, theta_0, 0, phi_f, steps)
        holonomy_values.append(holonomy)
    
    plt.figure()
    plt.plot(theta_values, holonomy_values, 'bo-', label='Holonomy Strength')
    plt.xlabel(r'$	heta_0$')
    plt.ylabel('Inner Product (Before vs. After Transport)')
    plt.title('Holonomy Strength for Different $Theta_0$')
    plt.legend()
    plt.savefig("holonomy.png")

# Example: Measure holonomy strength for different θ_0 values
theta_values = np.linspace(0.1, np.pi - 0.1, 20)  # Avoid singularities at poles
plot_holonomy_strength(theta_values)
