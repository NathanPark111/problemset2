import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Load the data using np.loadtxt
data = np.loadtxt('mesh.dat', skiprows=1)

# Extract the x and y coordinates
points = data

# Lifting function: z = x^2 + xy + y^2
def lifting_map(x, y):
    return x**2 + x * y + y**2

# Compute the first fundamental form (metric) at a point (x, y)
def first_fundamental_form(x, y):
    dz_dx = 2 * x + y  # Gradient of z = x^2 + xy + y^2 with respect to x
    dz_dy = x + 2 * y  # Gradient of z = x^2 + xy + y^2 with respect to y
    # First fundamental form is the metric tensor: [E, F; F, G]
    E = 1 + dz_dx**2
    F = dz_dx * dz_dy
    G = 1 + dz_dy**2
    return np.array([[E, F], [F, G]])

# Compute the second fundamental form (curvature tensor) at a point (x, y)
def second_fundamental_form(x, y):
    dz_dxx = 2  # Second derivative with respect to x
    dz_dxy = 1  # Mixed second derivative with respect to x and y
    dz_dyy = 2  # Second derivative with respect to y
    return np.array([[dz_dxx, dz_dxy], [dz_dxy, dz_dyy]])

# Compute the shape operator by diagonalizing the second fundamental form
def shape_operator(x, y):
    # First fundamental form (metric)
    G = first_fundamental_form(x, y)
    
    # Second fundamental form
    S = second_fundamental_form(x, y)
    
    # Shape operator is the inverse of the first fundamental form times the second fundamental form
    try:
        G_inv = np.linalg.inv(G)
        shape_op = np.dot(G_inv, S)
    except np.linalg.LinAlgError:
        shape_op = np.zeros((2, 2))  # In case the matrix is singular (flat region)
    
    # Diagonalize the shape operator to get principal curvatures
    eigenvalues, _ = np.linalg.eig(shape_op)
    k1, k2 = eigenvalues  # Principal curvatures
    return k1, k2

# Compute the curvatures for all points and store them
mean_curvature = np.zeros(len(points))
gaussian_curvature = np.zeros(len(points))
principal_curvatures_1 = np.zeros(len(points))
principal_curvatures_2 = np.zeros(len(points))

# List to store all the data for the text file
all_data = []

for i, point in enumerate(points):
    x, y = point
    k1, k2 = shape_operator(x, y)
    mean_curvature[i] = (k1 + k2) / 2
    gaussian_curvature[i] = k1 * k2
    principal_curvatures_1[i] = k1
    principal_curvatures_2[i] = k2
    
    # Store the data in a formatted string
    all_data.append(f"Point: ({x}, {y}) -> Mean Curvature: {mean_curvature[i]} -> "
                    f"Gaussian Curvature: {gaussian_curvature[i]} -> "
                    f"Principal Curvature (k1): {principal_curvatures_1[i]} -> "
                    f"Principal Curvature (k2): {principal_curvatures_2[i]}")

# Save all data into a single text file
with open("curvature_data2.txt", "w") as file:
    for line in all_data:
        file.write(line + "\n")

# Visualize the Mean Curvature as a heatmap
plt.figure(figsize=(10, 8))
plt.tricontourf(points[:, 0], points[:, 1], mean_curvature, cmap='jet')
plt.colorbar(label='Mean Curvature')
plt.scatter(points[:, 0], points[:, 1], color='black', label='Points')
plt.title('Mean Curvature')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('mean_curvature2.png')

# Visualize the Gaussian Curvature as a heatmap
plt.figure(figsize=(10, 8))
plt.tricontourf(points[:, 0], points[:, 1], gaussian_curvature, cmap='jet')
plt.colorbar(label='Gaussian Curvature')
plt.scatter(points[:, 0], points[:, 1], color='black', label='Points')
plt.title('Gaussian Curvature')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('gaussian_curvature2.png')

# Visualize the Principal Curvatures (k1) as a heatmap
plt.figure(figsize=(10, 8))
plt.tricontourf(points[:, 0], points[:, 1], principal_curvatures_1, cmap='jet')
plt.colorbar(label='Principal Curvature (k1)')
plt.scatter(points[:, 0], points[:, 1], color='black', label='Points')
plt.title('Principal Curvature (k1)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('principal_curvature_1_2.png')

# Visualize the Principal Curvatures (k2) as a heatmap
plt.figure(figsize=(10, 8))
plt.tricontourf(points[:, 0], points[:, 1], principal_curvatures_2, cmap='jet')
plt.colorbar(label='Principal Curvature (k2)')
plt.scatter(points[:, 0], points[:, 1], color='black', label='Points')
plt.title('Principal Curvature (k2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('principal_curvature_2_2.png')
