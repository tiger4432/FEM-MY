import numpy as np
from skfem import MeshTri
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def create_mesh_from_points(points):
    """Create a triangular mesh from 2D point cloud"""
    # Perform Delaunay triangulation
    tri = Delaunay(points.T)
    # Create mesh from triangulation
    return MeshTri(points, tri.simplices.T)

# Example: Create a point cloud with some structure
def generate_sample_points(n_points=100):
    """Generate a sample point cloud"""
    # Create base grid
    x = np.linspace(0, 1, int(np.sqrt(n_points)))
    y = np.linspace(0, 1, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)
    
    # Add some random perturbations
    X = X.flatten() + 0.1 * np.random.randn(n_points)
    Y = Y.flatten() + 0.1 * np.random.randn(n_points)
    
    return np.vstack((X, Y))

def visualize_mesh(mesh):
    """Visualize the mesh"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the mesh
    mesh.draw(ax=ax)
    
    # Plot the points
    ax.scatter(mesh.p[0], mesh.p[1], c='red', s=10)
    
    ax.set_title('Mesh from Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Generate sample points
    points = generate_sample_points()
    
    # Create mesh from points
    mesh = create_mesh_from_points(points)
    
    # Visualize
    visualize_mesh(mesh).show() 