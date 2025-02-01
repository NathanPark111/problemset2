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

# Compute the vertex normal (gradient of the surface)
def vertex_normal(x, y):
    # Gradient of z = x^2 + xy + y^2
    dz_dx = 2 * x + y  # derivative of x^2 + xy + y^2 with respect to x
    dz_dy = x + 2 * y  # derivative of x^2 + xy + y^2 with respect to y
    # Normal vector is (dz_dx, dz_dy, -1)
    return np.array([dz_dx, dz_dy, -1])

# Compute the magnitude of the normal vector
def normal_magnitude(x, y):
    normal = vertex_normal(x, y)
    return np.linalg.norm(normal)

# Lift the points to 3D using the lifting map
points_3d = np.column_stack((points, lifting_map(points[:, 0], points[:, 1])))

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

# List to store the vertex normal data for text output
normal_data = []

# Calculate and plot the vertex normals at each vertex
for point in points:
    x, y = point
    normal = vertex_normal(x, y)
    magnitude = normal_magnitude(x, y)
    print(f"Point ({x}, {y}) -> Normal Magnitude: {magnitude}")  # Print normal magnitude
    
    # Manually scale the normal vector to fit better in the plot
    normal_scaled = normal / magnitude * 0.5  # Adjusted scaling factor
    
    # Plot the normal vectors as arrows with scaled size
    ax.quiver(x, y, normal_scaled[0], normal_scaled[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.005)
    
    # Add the vector normal to the normal data list for saving
    normal_data.append(f"Point: ({x}, {y}) -> Vertex Normal: ({normal[0]}, {normal[1]})")

# Save the vertex normal data to a text file
with open("vertex_normals2.txt", "w") as file:
    for line in normal_data:
        file.write(line + "\n")

# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Mesh with Vertex Normals')

# Show the plot
plt.legend()
plt.savefig("vertex_normal2.png")
plt.show()
