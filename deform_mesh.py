import numpy as np
from skfem import *
from skfem.helpers import dot
import matplotlib.pyplot as plt

def create_rectangular_mesh(width=5, height=1, nx=50, ny=10):
    """Create a rectangular mesh with denser points at the top and sparser at the bottom"""
    # Create x-axis points (uniform)
    m1 = MeshLine(np.linspace(0, width, nx))
    
    # Create y-axis points (non-uniform, denser at top)
    # Use exponential function to create denser points at the top
    y_points = np.linspace(0, 1, ny)
    y_points = height * (1 - np.exp(-3 * y_points))  # Exponential distribution
    m2 = MeshLine(y_points)
    
    return (m1 * m2).with_defaults()

def get_grid_structure(mesh):
    """Get the grid structure (nx, ny) of a refined mesh
    
    Args:
        mesh: The refined mesh
        
    Returns:
        tuple: (nx, ny) where nx is the number of points in x-direction
              and ny is the number of points in y-direction
    """
    # Get unique x and y coordinates
    unique_x = np.unique(mesh.p[0])
    unique_y = np.unique(mesh.p[1])
    
    # Count number of unique points in each direction
    nx = len(unique_x)
    ny = len(unique_y)
    
    return nx, ny

def create_and_refine_mesh(width=5, height=1, nx=50, ny=10, refine_level=1):
    """Create and refine a rectangular mesh, returning both the mesh and its grid structure"""
    # Create initial mesh
    mesh = create_rectangular_mesh(width, height, nx, ny)
    
    # Refine the mesh
    for _ in range(refine_level):
        mesh = mesh.refined()
    
    # Get the grid structure after refinement
    refined_nx, refined_ny = get_grid_structure(mesh)
    
    return mesh, (refined_nx, refined_ny)

def calculate_contact_area(mesh1, mesh2, tolerance=1e-6):
    """Calculate the contact area between two meshes"""
    # Get nodes from both meshes
    nodes1 = mesh1.p.T
    nodes2 = mesh2.p.T
    
    # Calculate distances between nodes
    distances = np.zeros((len(nodes1), len(nodes2)))
    for i, n1 in enumerate(nodes1):
        for j, n2 in enumerate(nodes2):
            distances[i,j] = np.linalg.norm(n1 - n2)
    
    # Find contact nodes
    contact_nodes = distances < tolerance
    
    # Calculate contact area
    contact_area = np.sum(contact_nodes) * (mesh1.p[0,1] - mesh1.p[0,0])  # Approximate area
    
    return contact_area, contact_nodes

def deform_top_surface(mesh, top_deformations, amplitude=0.5, contact_mesh=None):
    """Deform the mesh while preserving vertical ordering and fixing the bottom surface
    
    Args:
        mesh: The mesh to deform
        top_deformations: Array of target deformations for each point on the top surface
        amplitude: Maximum deformation amplitude
        contact_mesh: Optional mesh for contact calculation
    """
    # Get unique y-coordinates and sort them
    y_coords = np.unique(mesh.p[1])
    n_layers = len(y_coords)
    
    # Apply deformation to all layers except the bottom
    for i, y in enumerate(y_coords[:-1]):  # Exclude the bottom layer
        # Find nodes in this layer
        layer_nodes = np.where(mesh.p[1] == y)[0]
        
        # Calculate deformation for this layer
        # Amplitude decreases linearly from top to bottom
        # The bottom layer (i = n_layers-2) will have zero deformation
        layer_amplitude = (i + 1) / (n_layers - 1)  # Adjusted for bottom layer exclusion
        deformation = top_deformations * layer_amplitude
        
        # Apply deformation
        mesh.p[1, layer_nodes] += deformation
        
        # If contact mesh is provided, calculate and update contact area
        if contact_mesh is not None and i == 0:  # Only for top layer
            contact_area, _ = calculate_contact_area(mesh, contact_mesh)
            print(f"Current contact area: {contact_area:.6f}")

def visualize_mesh(mesh, title="Mesh"):
    """Visualize the mesh"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the mesh
    mesh.draw(ax=ax)
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create and refine mesh
    mesh, (refined_nx, refined_ny) = create_and_refine_mesh(refine_level=1)
    print(f"Grid structure after refinement: nx={refined_nx}, ny={refined_ny}")
    
    # Create contact mesh (flat surface)
    contact_mesh = create_rectangular_mesh(height=0.1, ny=2)  # Flat surface
    contact_mesh.p[1] += 1.0  # Move up for initial contact
    
    # Visualize original mesh
    fig1 = visualize_mesh(mesh, "Original Mesh")
    fig1.show()
    
    # Get top surface points
    y_coords = np.unique(mesh.p[1])
    top_nodes = np.where(mesh.p[1] == y_coords[-1])[0]
    x_coords = mesh.p[0, top_nodes]
    
    # Create custom deformations for top surface
    # Example: Multiple sine waves with different frequencies
    top_deformations = (
        0.2 * np.sin(2 * np.pi * x_coords / mesh.p[0].max()) +
        0.1 * np.sin(4 * np.pi * x_coords / mesh.p[0].max()) +
        0.05 * np.sin(8 * np.pi * x_coords / mesh.p[0].max())
    )
    
    # Deform mesh while preserving vertical ordering and fixing the bottom
    deform_top_surface(mesh, 
                      top_deformations=top_deformations,
                      amplitude=0.3,
                      contact_mesh=contact_mesh)
    
    # Visualize deformed mesh
    fig2 = visualize_mesh(mesh, "Deformed Mesh (Fixed Bottom)")
    fig2.show()