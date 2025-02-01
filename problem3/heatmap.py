import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import matplotlib.colors as mcolors

# Load the data using np.loadtxt
data = np.loadtxt('mesh.dat', skiprows=1)

# Extract the x and y coordinates
points = data

# Lifting function
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

# Lift the points to 3D using the map z = x^2 + y^2
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

# Convert area ratios to a numpy array
area_ratios = np.array(area_ratios)

# Adjust the normalization to range from 0 to 50 for the heatmap
norm = mcolors.Normalize(vmin=0, vmax=50)
cmap = plt.cm.viridis

# Plot the heatmap by filling the triangles with color based on area ratio
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each triangle, filling the area with color based on area ratio
for i, simplex in enumerate(triangulation.simplices):
    p0, p1, p2 = points[simplex]
    # Compute the area ratio for this triangle
    area_ratio = area_ratios[i]
    
    # Define the triangle vertices
    triangle = np.array([p0, p1, p2, p0])
    
    # Plot the filled triangle
    ax.fill(triangle[:, 0], triangle[:, 1], color=cmap(norm(area_ratio)), edgecolor='k', alpha=0.7)

# Add the colorbar using the colormap and normalization
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array for the colorbar
fig.colorbar(sm, ax=ax, label='Area Ratio (After / Before)')

# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Heatmap of Area Ratios (After Lifting to 3D)')

# Show the plot
plt.savefig("heatmap2.png")
