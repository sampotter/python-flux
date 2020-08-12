cmap = dict()
try:
    import colorcet as cc
    cmap['jet'] = cc.cm.rainbow
    cmap['gray'] = cc.cm.gray
    cmap['fire'] = cc.cm.fire
except ImportError:
    print('failed to import colorcet: using matplotlib colormaps')
    cmap['jet'] = 'jet'
    cmap['gray'] = 'gray'
    cmap['fire'] = 'inferno'

try:
    import distmesh as dm
    USE_DISTMESH = True
except ImportError:
    print('failed to import distmesh: will use scipy.spatial.Delaunay')
    USE_DISTMESH = False

import embree
import matplotlib.pyplot as plt
import meshio
import netCDF4
import numpy as np
import os
import scipy.interpolate

from form_factors import FormFactorMatrix, _compute_FF_block

from scipy.constants import sigma # Stefan-Boltzmann constant

def get_centroids(V, F):
    return V[F].mean(axis=1)

def get_cross_products(V, F):
    V0 = V[F][:, 0, :]
    C = np.cross(V[F][:, 1, :] - V0, V[F][:, 2, :] - V0)
    return C

def get_surface_normals_and_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    A = C_norms/2
    return N, A

def solve_kernel_system(L, b, rho=1, tol=np.finfo(np.float64).eps):
    x = b
    dx = L@(rho*b)
    nmul = 1
    err = np.linalg.norm(b - x + dx)
    while np.linalg.norm(err) > tol:
        x = b + dx
        dx = L@(rho*b)
        nmul += 1
        prev_err = err
        err = np.linalg.norm(b - x + dx)
        if abs(err - prev_err) < tol:
            break
    return x, nmul

def compute_T(L, Q, rho, emiss, tol=np.finfo(np.float64).eps):
    Q, nmul1 = solve_kernel_system(L, Q, rho, tol)
    Q *= 1 - rho
    tmp, nmul2 = solve_kernel_system(L, Q, 1, tol)
    Q = (1 - emiss)*Q + emiss*tmp
    Q /= emiss*sigma
    Q = Q**(1/4)
    return Q, nmul1 + nmul2

def tripcolor_vector(V, F, v, I=None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    if I is None:
        im = ax.tripcolor(*V[:, :2].T, F, v, **kwargs)
    else:
        im = ax.tripcolor(*V[:, :2].T, F[I], v[I], **kwargs)
    fig.colorbar(im, ax=ax)
    ax.set_aspect('equal')
    xmin, ymin = np.min(V[:, :2], axis=0)
    xmax, ymax = np.max(V[:, :2], axis=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    return fig, ax

if __name__ == '__main__':
    # Define constants used in the simulation:
    e0 = 10*np.pi/180 # Solar elevation angle
    F0 = 1365 # Solar constant
    emiss = 0.95 # Emissitivity
    rho = 0.12 # Visual (?) albedo
    dir_sun = np.array([0, -np.cos(e0), np.sin(e0)]) # Direction of sun
    x0, x1, y0, y1 = -35, 25, 40, 100 # Bounding box of crater (not
                                      # used except in
                                      # lunar_south_pole.png plot at
                                      # the moment)
    h = 1.5 # Desired edge lengths of triangles generated by DistMesh
    p0 = np.array([-5, 75]) # Center of circular mesh
    r0 = 25 # # Radius of circular mesh

    # First, load the DEM of the lunar south pole, which is stored as
    # a netCDF4 file, and pull out the coordinate data.
    path = os.path.join('.', 'lunar_south_pole_80mpp_curvature.grd')
    rootgrp = netCDF4.Dataset(path)
    X = np.array(rootgrp.variables['x'])
    Y = np.array(rootgrp.variables['y'])
    Z = np.array(rootgrp.variables['z'])
    print('loaded lunar_south_pole_80mpp_curvature.grd')

    # As a quick sanity check and to show some basics of how to plot
    # using matplotlib, let's plot the DEM, draw a bounding box around
    # the crater, and save the plot to disk as a PNG.
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(Z, extent=[X.min(), X.max(), Y.max(), X.min()],
                   cmap=cmap['jet'], zorder=1)
    ax.set_aspect('equal')
    ax.grid(zorder=2)
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c='k',
            linewidth=1, linestyle='--', zorder=3)
    ax.set_xlabel('$x$ [km]')
    ax.set_ylabel('$y$ [km]')
    ax.set_title('Elevation ($z$) [km]')
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax)
    fig.savefig('lunar_south_pole.png')
    plt.close(fig)
    print('wrote lunar_south_pole.png to disk')

    # Create function z = z(x, y) that linearly interpolates DEM data
    z = scipy.interpolate.interp2d(X, Y, Z)

    if USE_DISTMESH:
        # Here, we use DistMesh to mesh Haworth. We use the library
        # PyDistmesh which reimplements distmesh in Python. This snippet
        # of code could modified without too much trouble to mesh the
        # entire rectangular bounding box, and to vary the target edge
        # length ("fh" below).
        fd = lambda p: dm.dcircle(p, *p0, r0)
        fh = dm.huniform
        V, F = dm.distmesh2d(fd, fh, h, (x0, x1, y0, y1), fig=None)
    else:
        X_mesh, Y_mesh = np.meshgrid(
            np.linspace(x0, x1, int((x1 - x0)//h)),
            np.linspace(y0, y1, int((y1 - y0)//h))
        )
        points_mesh = np.array([X_mesh.flatten(), Y_mesh.flatten()]).T
        delaunay = scipy.spatial.Delaunay(points_mesh)
        V, F = delaunay.points, delaunay.simplices
    V = np.row_stack([V.T, np.array([z(*v)[0] for v in V])]).T
    num_faces = F.shape[0]
    print('created mesh with %d triangles' % num_faces)

    # Let's use another Python library (meshio) to save the triangle
    # mesh generated by Distmesh as an OBJ file.
    points = V
    cells = [('triangle', F)]
    mesh = meshio.Mesh(points, cells)
    mesh_path = 'haworth%d.obj' % num_faces
    mesh.write(mesh_path)
    print('wrote %s to disk' % mesh_path)

    # To build our form factor matrix, we first need to get the
    # centroids (P), face normals (N), triangle areas (A).
    P = get_centroids(V, F)
    N, A = get_surface_normals_and_areas(V, F)

    # Next, we need to set up Embree. The lines below allocate some
    # memory that Embree manages, and loads our vertices and index
    # lists for the faces. In Embree parlance, we create a "device",
    # which manages a "scene", which has one "geometry" in it, which
    # is our mesh.
    device = embree.Device()
    geometry = device.make_geometry(embree.GeometryType.Triangle)
    scene = device.make_scene()
    vertex_buffer = geometry.set_new_buffer(
        embree.BufferType.Vertex, # buf_type
        0, # slot
        embree.Format.Float3, # fmt
        3*np.dtype('float32').itemsize, # byte_stride
        V.shape[0], # item_count
    )
    vertex_buffer[:] = V[:]
    index_buffer = geometry.set_new_buffer(
        embree.BufferType.Index, # buf_type
        0, # slot
        embree.Format.Uint3, # fmt
        3*np.dtype('uint32').itemsize, # byte_stride,
        F.shape[0]
    )
    index_buffer[:] = F[:]
    geometry.commit()
    scene.attach_geometry(geometry)
    geometry.release()
    scene.commit()

    # Build the compressed form factor matrix. All of the code related
    # to this can be found in the "form_factors.py" file in this
    # directory.
    FF = FormFactorMatrix.assemble_using_quadtree(scene, V, F, tol=5e-4)

    # Python makes it very easy to serialize object hierarchies and
    # write them to disk as binary files. We do that here to save the
    # compressed form factor matrix. We can reload it later if we
    # want, without having to first compute it (or load an OBJ file,
    # or set up Embree, or any of that stuff).
    FF.save('FF.bin')
    print('saved compressed form factor matrix to FF.bin')

    # We can also build the uncompressed form factor matrix, which
    # would be done like this:
    ### FF_gt = _compute_FF_block(P, N, A,
    ### scene=scene) FF_gt = scipy.sparse.csr_matrix(FF_gt)

    # Here, we use Embree directly to fine the indices of triangles
    # which are directly illuminated (I_sun) or not (I_shadow).
    eps = 1e3*np.finfo(np.float32).resolution
    ray = embree.Ray1M(num_faces)
    ray.org[:] = P + eps*N
    ray.dir[:] = dir_sun
    ray.tnear[:] = 0
    ray.tfar[:] = np.inf
    ray.flags[:] = 0
    context = embree.IntersectContext()
    scene.occluded1M(context, ray)
    I_sun = np.isposinf(ray.tfar)
    I_shadow = np.logical_not(I_sun)
    print('found illuminated region')

    # Compute the direct illumination.
    E = np.zeros((N.shape[0],))
    E[I_sun] = F0*(N[I_sun]@dir_sun)
    print('computed direct illumination')

    # Compute the steady state temperature. This function is at the
    # top of this file.
    T, nmul = compute_T(FF, E, rho, emiss)
    print('computed T (%d matrix multiplications)' % (nmul,))

    # Finally, we make some plots showing what we just did, and write
    # them to disk:

    fig, ax = FF.show()
    fig.savefig('haworth_blocks.png')
    plt.close(fig)
    print('wrote haworth_blocks.png to disk')

    fig, ax = tripcolor_vector(V, F, E, cmap=cmap['gray'])
    fig.savefig('haworth_E.png')
    plt.close(fig)
    print('wrote haworth_E.png to disk')

    fig, ax = tripcolor_vector(V, F, T, vmin=0, vmax=400, cmap=cmap['fire'])
    fig.savefig('haworth_T.png')
    plt.close(fig)
    print('wrote haworth_T.png to disk')

    fig, ax = tripcolor_vector(V, F, T, I=I_shadow, cmap=cmap['jet'])
    fig.savefig('haworth_T_shadow.png')
    plt.close(fig)
    print('wrote haworth_T_shadow.png to disk')
