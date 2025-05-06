import numpy as np
from skfem import MeshTri
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

def create_mesh_from_boundary(boundary_points, density=0.1):
    """Create a mesh from boundary points
    
    Parameters:
    -----------
    boundary_points : np.ndarray
        Array of shape (2, n) containing x,y coordinates of boundary points
    density : float
        Controls the density of internal points (smaller = more points)
    """
    # Create polygon from boundary points
    polygon = Polygon(boundary_points.T)
    
    # Create a slightly smaller polygon to ensure points are strictly inside
    inner_polygon = polygon.buffer(-density/2)
    
    # Get bounding box of the inner polygon
    minx, miny, maxx, maxy = inner_polygon.bounds
    
    # Generate grid of points
    x = np.arange(minx, maxx, density)
    y = np.arange(miny, maxy, density)
    X, Y = np.meshgrid(x, y)
    
    # Flatten and stack coordinates
    points = np.vstack((X.flatten(), Y.flatten()))
    
    # Filter points inside polygon
    inside = []
    for i in range(points.shape[1]):
        if inner_polygon.contains(Point(points[:, i])):
            inside.append(points[:, i])
    
    # Combine boundary and internal points
    all_points = np.hstack((boundary_points, np.array(inside).T))
    
    # Create mesh using Delaunay triangulation
    tri = Delaunay(all_points.T)
    mesh = MeshTri(all_points, tri.simplices.T)
    
    # Manually specify boundary facets
    n_boundary = boundary_points.shape[1]
    boundaries = {}
    
    # Create boundary facets for each side
    for i in range(n_boundary):
        # Find indices of boundary points in all_points
        idx1 = np.where((all_points[0] == boundary_points[0,i]) & 
                       (all_points[1] == boundary_points[1,i]))[0][0]
        idx2 = np.where((all_points[0] == boundary_points[0,(i+1)%n_boundary]) & 
                       (all_points[1] == boundary_points[1,(i+1)%n_boundary]))[0][0]
        
        # Add to boundaries dictionary
        boundaries[f'boundary_{i}'] = np.array([idx1, idx2])
    
    # Set boundaries in mesh
    mesh.boundaries = boundaries
    
    return mesh

def generate_circle_boundary(n_points=50, radius=1.0):
    """Generate points on a circle"""
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.vstack((x, y))

def generate_rectangle_boundary(width=2.0, height=1.0, n_points_per_side=20):
    """Generate points on a rectangle"""
    # Generate points for each side
    x_bottom = np.linspace(-width/2, width/2, n_points_per_side)
    y_bottom = np.full_like(x_bottom, -height/2)
    
    x_right = np.full(n_points_per_side, width/2)
    y_right = np.linspace(-height/2, height/2, n_points_per_side)
    
    x_top = np.linspace(width/2, -width/2, n_points_per_side)
    y_top = np.full_like(x_top, height/2)
    
    x_left = np.full(n_points_per_side, -width/2)
    y_left = np.linspace(height/2, -height/2, n_points_per_side)
    
    # Combine all points
    x = np.concatenate([x_bottom, x_right, x_top, x_left])
    y = np.concatenate([y_bottom, y_right, y_top, y_left])
    
    return np.vstack((x, y))

def generate_rough_top_boundary(width=2.0, height=1.0, n_points_per_side=20, roughness=0.2):
    """Generate points for a shape with flat bottom and rough top
    
    Parameters:
    -----------
    width : float
        Width of the shape
    height : float
        Height of the shape
    n_points_per_side : int
        Number of points per side
    roughness : float
        Amplitude of the roughness (0 to 1)
    """
    # Bottom side (flat)
    x_bottom = np.linspace(-width/2, width/2, n_points_per_side)
    y_bottom = np.full_like(x_bottom, -height/2)
    
    # Right side (straight)
    x_right = np.full(n_points_per_side, width/2)
    y_right = np.linspace(-height/2, height/2, n_points_per_side)
    
    # Top side (rough)
    x_top = np.linspace(width/2, -width/2, n_points_per_side)
    # Add roughness using sine waves with different frequencies
    y_top = height/2 + roughness * height * (
        0.5 * np.sin(4 * np.pi * x_top / width) +
        0.3 * np.sin(8 * np.pi * x_top / width) +
        0.2 * np.random.randn(n_points_per_side)
    )
    
    # Left side (straight)
    x_left = np.full(n_points_per_side, -width/2)
    y_left = np.linspace(height/2, -height/2, n_points_per_side)
    
    # Combine all points
    x = np.concatenate([x_bottom, x_right, x_top, x_left])
    y = np.concatenate([y_bottom, y_right, y_top, y_left])
    
    return np.vstack((x, y))

def visualize_mesh(mesh, boundary_points=None):
    """Visualize the mesh"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the mesh
    mesh.draw(ax=ax)
    
    # Plot boundary points if provided
    if boundary_points is not None:
        ax.scatter(boundary_points[0], boundary_points[1], 
                  c='red', s=50, label='Boundary Points')
    
    # Plot boundary facets
    if hasattr(mesh, 'boundaries'):
        for name, facet in mesh.boundaries.items():
            ax.plot(mesh.p[0, facet], mesh.p[1, facet], 'r-', linewidth=2)
    
    ax.set_title('Mesh from Boundary Points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    if boundary_points is not None:
        ax.legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example 1: Circle
    boundary_points = generate_circle_boundary()
    mesh = create_mesh_from_boundary(boundary_points, density=0.2)
    visualize_mesh(mesh, boundary_points).show()
    
    # Example 2: Rectangle
    boundary_points = generate_rectangle_boundary()
    mesh = create_mesh_from_boundary(boundary_points, density=0.2)
    visualize_mesh(mesh, boundary_points).show()
    
    # Example 3: Rough top shape
    boundary_points = generate_rough_top_boundary(roughness=0.3)
    mesh = create_mesh_from_boundary(boundary_points, density=0.2)
    visualize_mesh(mesh, boundary_points).show() 