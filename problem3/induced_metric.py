import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import matplotlib.colors as mcolors

# Load the data using np.loadtxt
data = np.loadtxt('mesh.dat', skiprows=1)

# Extract the x and y coordinates
points = data

# Lifting function: z = x^2 + xy + y^2
def lifting_map(x, y):
    return x**2 + x*y + y**2

# Function to compute area of a triangle in 2D
def area_2d(a, b, c):
    return 0.5 * np.abs(a[0]*(b[1] - c[1]) + b[0]*(c[1] - a[1]) + c[0]*(a[1] - b[1]))

# Function to compute area of a triangle in 3D
def area_3d(a, b, c):
    # Vectors representing two edges of the triangle
    v1 = b - a
    v2 = c - a
    # Cross product of the two vectors
    cross_product = np.cross(v1, v2)
    return 0.5 * np.linalg.norm(cross_product)

# Lift the points to 3D using the map z = x^2 + xy + y^2
points_3d = np.column_stack((points, lifting_map(points[:, 0], points[:, 1])))

# Generate the convex hull
hull = ConvexHull(points)

# Generate the Delaunay triangulation
triangulation = Delaunay(points)

# Initialize a list to store area ratios
area_ratios = []

# Loop over each triangle in the Delaunay triangulation
for simplex in triangulation.simplices:
    # 2D points for the triangle
    p0, p1, p2 = points[simplex]
    
    # Compute area in 2D
    area_2d_before = area_2d(p0, p1, p2)
    
    # 3D points for the triangle after lifting
    p0_3d, p1_3d, p2_3d = points_3d[simplex]
    
    # Compute area in 3D
    area_3d_after = area_3d(p0_3d, p1_3d, p2_3d)
    
    # Calculate the area ratio
    area_ratio = area_3d_after / area_2d_before
    area_ratios.append(area_ratio)

# Function to compute the induced metric for the surface z = x^2 + xy + y^2
def induced_metric(x, y):
    dz_dx = 2*x + y  # Partial derivative of z with respect to x
    dz_dy = x + 2*y  # Partial derivative of z with respect to y
    g11 = 1 + dz_dx**2
    g12 = g21 = dz_dx * dz_dy
    g22 = 1 + dz_dy**2
    return np.array([[g11, g12], [g21, g22]])

# Write the results to a text file
with open("induced_metric_results2.txt", "w") as file:
    # Write headers
    file.write("Point (x, y) -> Induced Metric Matrix\n")
    file.write("----------------------------------------\n")
    
    for point in points:
        x, y = point
        g = induced_metric(x, y)
        file.write(f"Point ({x}, {y}):\n")
        file.write(f"{g}\n\n")

print("Induced metric results have been saved to 'induced_metric_results.txt'.")
