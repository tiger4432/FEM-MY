r"""Contact problem.

Mortar methods allow setting interface conditions on non-matching meshes.
They are useful also when solving variational inequalities such as
`elastic contact problems <https://arxiv.org/abs/1902.09312>`_.

"""

import numpy as np
from skfem import *
from skfem.supermeshing import intersect, elementwise_quadrature
from skfem.models.elasticity import (linear_elasticity, lame_parameters,
                                     linear_stress)
from skfem.helpers import dot, sym_grad, jump, mul
from skfem.io.json import from_file
from pathlib import Path


# create meshes
mesh_file =  'ex04_mesh.json'
m1 = from_file(mesh_file)

# Create a rough surface profile in 2D
def create_rough_profile(n_points=100, amplitude=0.1, wavelength=0.1):
    """Create a 2D rough surface profile"""
    y = np.linspace(-1, 1, n_points)
    # Add random perturbations with controlled frequency
    x = 1.0 + amplitude * np.sin(2 * np.pi * y / wavelength) + 0.05 * np.random.randn(n_points)
    return np.vstack((x, y))

# Create base points for the rough profile
points = create_rough_profile()
# Create mesh from points
m2 = MeshLine(points).refined().with_boundaries({
    'contact': lambda x: x[0] == 1.0,
    'dirichlet': lambda x: x[0] == 2.0,
})

e1 = ElementVector(ElementTriP2())
e2 = ElementVector(ElementQuad2())

# create trace meshes and project
m1t, orig1 = m1.trace('contact', mtype=MeshLine, project=lambda p: p[1:])
m2t, orig2 = m2.trace('contact', mtype=MeshLine, project=lambda p: p[1:])

# create a supermesh for integration
m12, t1, t2 = intersect(m1t, m2t)

basis1 = Basis(m1, e1)
basis2 = Basis(m2, e2)

fbasis1 = FacetBasis(m1, e1,
                     quadrature=elementwise_quadrature(m1t, m12, t1),
                     facets=orig1[t1])
fbasis2 = FacetBasis(m2, e2,
                     quadrature=elementwise_quadrature(m2t, m12, t2),
                     facets=orig2[t2])
fbasis = fbasis1 * fbasis2

# problem definition
youngs_modulus = 1000.0
poisson_ratio = 0.3

weakform = linear_elasticity(*lame_parameters(youngs_modulus, poisson_ratio))
C = linear_stress(*lame_parameters(youngs_modulus, poisson_ratio))

alpha = 1000
limit = 0.3

# mortar forms
@BilinearForm
def bilin_mortar(u1, u2, v1, v2, w):
    ju = dot(u1 - u2, w.n)
    jv = dot(v1 - v2, w.n)
    mu = .5 * (dot(w.n, mul(C(sym_grad(u1)), w.n))
               + dot(w.n, mul(C(sym_grad(u2)), w.n)))
    mv = .5 * (dot(w.n, mul(C(sym_grad(v1)), w.n))
               + dot(w.n, mul(C(sym_grad(v2)), w.n)))
    return ((1. / (alpha * w.h) * ju * jv - mu * jv - mv * ju)
            * (np.abs(w.x[1]) <= limit))

def gap(x):
    return (1. - np.sqrt(1. - x[1] ** 2))

@LinearForm
def lin_mortar(v1, v2, w):
    jv = dot(v1 - v2, w.n)
    mv = .5 * (dot(w.n, mul(C(sym_grad(v1)), w.n))
               + dot(w.n, mul(C(sym_grad(v2)), w.n)))
    return ((1. / (alpha * w.h) * gap(w.x) * jv - gap(w.x) * mv)
            * (np.abs(w.x[1]) <= limit))

# fix mesh parameter and normals from m2
params = {
    'h': fbasis2.mesh_parameters(),
    'n': -fbasis2.normals,
}

# assemble the block system
A1 = asm(weakform, basis1)
A2 = asm(weakform, basis2)
B = asm(bilin_mortar, fbasis, **params)
f = asm(lin_mortar, fbasis, **params)

K = bmat([[A1, None],
          [None, A2]], 'csr') + B

D1 = basis1.get_dofs('dirichlet').all()
D2 = basis2.get_dofs('dirichlet').all() + basis1.N

# initialize boundary conditions
y1 = basis1.zeros()
y2 = basis2.zeros()
y1[basis1.get_dofs('dirichlet').all('u^1')] = .1
y = np.concatenate((y1, y2))

# linear solve
y = solve(*condense(K, f, D=np.concatenate((D1, D2)), x=y))

# create a displaced mesh for visualization
sf = 1
(y1, _), (y2, _) = fbasis.split(y)
mdefo1 = m1.translated(sf * y1[basis1.nodal_dofs])
mdefo2 = m2.translated(sf * y2[basis2.nodal_dofs])

# calculate von Mises stress
s1, s2 = {}, {}
dg1 = basis1.with_element(ElementTriDG(ElementTriP1()))
dg2 = basis2.with_element(ElementQuadDG(ElementQuad1()))
u1 = basis1.interpolate(y1)
u2 = basis2.interpolate(y2)

for i in [0, 1]:
    for j in [0, 1]:
        s1[i, j] = dg1.project(C(sym_grad(u1))[i, j])
        s2[i, j] = dg2.project(C(sym_grad(u2))[i, j])

s1[2, 2] = poisson_ratio * (s1[0, 0] + s1[1, 1])
s2[2, 2] = poisson_ratio * (s2[0, 0] + s2[1, 1])

vonmises1 = np.sqrt(.5 * ((s1[0, 0] - s1[1, 1]) ** 2 +
                          (s1[1, 1] - s1[2, 2]) ** 2 +
                          (s1[2, 2] - s1[0, 0]) ** 2 +
                          6. * s1[0, 1] ** 2))

vonmises2 = np.sqrt(.5 * ((s2[0, 0] - s2[1, 1]) ** 2 +
                          (s2[1, 1] - s2[2, 2]) ** 2 +
                          (s2[2, 2] - s2[0, 0]) ** 2 +
                          6. * s2[0, 1] ** 2))


def visualize():
    from skfem.visuals.matplotlib import plot, draw
    pdg1 = Basis(mdefo1, dg1.elem)
    pdg2 = Basis(mdefo2, dg2.elem)
    ax = plot(pdg1,
              vonmises1,
              shading='gouraud',
              colorbar=r"$\sigma_{\mathrm{mises}}$")
    draw(mdefo1, ax=ax)
    plot(pdg2, vonmises2, ax=ax, nrefs=3, shading='gouraud')
    draw(mdefo2, ax=ax)
    return ax

def visualize_mesh_trace():
    """Visualize the relationship between m1 and its traced contact surface m1t"""
    import matplotlib.pyplot as plt
    from skfem.visuals.matplotlib import draw
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original mesh m1
    draw(m1, ax=ax1)
    ax1.set_title('Original Mesh m1')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Add axis values for m1
    x_min, x_max = m1.p[0].min(), m1.p[0].max()
    y_min, y_max = m1.p[1].min(), m1.p[1].max()
    ax1.set_xticks(np.linspace(x_min, x_max, 5))
    ax1.set_yticks(np.linspace(y_min, y_max, 5))
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot traced contact surface m1t
    draw(m1t, ax=ax2)
    ax2.set_title('Traced Contact Surface m1t')
    ax2.set_xlabel('y')  # Note: m1t is 1D, so we only have y-coordinate
    ax2.set_ylabel('z')
    
    # Add axis values for m1t
    y_min_t, y_max_t = m1t.p[0].min(), m1t.p[0].max()
    ax2.set_xticks(np.linspace(y_min_t, y_max_t, 5))
    ax2.set_yticks([-0.1, 0, 0.1])  # z-axis range for 1D mesh
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_rough_surface():
    """Visualize the 2D rough surface profile"""
    import matplotlib.pyplot as plt
    from skfem.visuals.matplotlib import draw
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the mesh
    draw(m2, ax=ax)
    
    # Plot the profile line
    x = m2.p[0]
    y = m2.p[1]
    ax.plot(x, y, 'r-', linewidth=2, alpha=0.5)
    
    # Add scatter points for mesh nodes
    ax.scatter(x, y, c='b', s=10)
    
    ax.set_title('2D Rough Surface Profile')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # visualize().show()
    # visualize_mesh_trace().show()
    visualize_rough_surface().show()