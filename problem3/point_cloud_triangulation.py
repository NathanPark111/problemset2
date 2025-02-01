import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import matplotlib.colors as mcolors

# Load the data using np.loadtxt
data = np.loadtxt('mesh.dat', skiprows=1)

# Extract the x and y coordinates
points = data

# Generate the convex hull
hull = ConvexHull(points)

# Generate the Delaunay triangulation
triangulation = Delaunay(points)

# Plot the convex hull
plt.figure(figsize=(8, 6))

# Plot the Delaunay triangulation
plt.triplot(points[:, 0], points[:, 1], triangulation.simplices, color='blue', alpha=0.5)

# Plot the convex hull
plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r-', lw=2, label='Convex Hull')

# Scatter plot the points
plt.scatter(points[:, 0], points[:, 1], color='black', zorder=5, label='Data Points')

# Adding labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Convex Hull and Delaunay Triangulation')
plt.legend()

# Show the plot
plt.savefig("point_cloud_triangulation.png")

