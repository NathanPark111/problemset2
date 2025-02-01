import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Load the data using np.loadtxt
data = np.loadtxt('mesh.dat', skiprows=1)

# Extract the x and y coordinates
points = data

# Lifting function: z = x^2 + xy + y^2
def lifting_map(x, y):
    return x**2 + x*y + y**2

# Compute the normal of the surface (face normal) for a triangle
def face_normal(x1, y1, x2, y2, x3, y3):
    # Lift points to 3D using z = x^2 + xy + y^2
    z1, z2, z3 = lifting_map(x1, y1), lifting_map(x2, y2), lifting_map(x3, y3)
    
    # Create vectors for the edges of the triangle
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x3 - x1, y3 - y1, z3 - z1])
    
    # Compute the cross product to get the normal vector
    normal = np.cross(v1, v2)
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    return normal

# Generate the Delaunay triangulation
triangulation = Delaunay(points)

# Create a plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Plot the mesh points
ax.scatter(points[:, 0], points[:, 1], color='black', zorder=5, label='Data Points')

# Plot the Delaunay triangulation
for simplex in triangulation.simplices:
    ax.plot(points[simplex, 0], points[simplex, 1], color='blue', alpha=0.5)

# Manually scale the normal vectors for visibility
normal_scale = 1  # Set this value to control the vector length

# Create a list to store the normal data
normal_data = []

# Calculate and plot the surface normals for each triangle (face normal)
for simplex in triangulation.simplices:
    # Get the coordinates of the three vertices of the triangle
    x1, y1 = points[simplex[0]]
    x2, y2 = points[simplex[1]]
    x3, y3 = points[simplex[2]]
    
    # Compute the surface normal for the triangle
    normal = face_normal(x1, y1, x2, y2, x3, y3)
    
    # Plot the normal vectors as arrows
    # The arrow will be placed at the centroid of the triangle
    centroid_x = (x1 + x2 + x3) / 3
    centroid_y = (y1 + y2 + y3) / 3
    
    # Scale the normal vector manually to a fixed length for visibility
    ax.quiver(centroid_x, centroid_y, normal[0] * normal_scale, normal[1] * normal_scale, angles='xy', scale_units='xy', scale=1, color='red')
    
    # Add the triangle vertices and normal to the list
    normal_data.append(f"Triangle: ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}) -> Normal: ({normal[0]}, {normal[1]})")

# Save the normal data to a text file
with open("surface_normals2.txt", "w") as file:
    for line in normal_data:
        file.write(line + "\n")

# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Mesh with Surface Normals')

# Manually set axis limits to ensure normal vectors are visible
ax.set_xlim([np.min(points[:, 0]) - 1, np.max(points[:, 0]) + 1])
ax.set_ylim([np.min(points[:, 1]) - 1, np.max(points[:, 1]) + 1])

# Show the plot
plt.legend()
plt.savefig("surface_normal2.png")
plt.show()
