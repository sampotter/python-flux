import embree
import matplotlib.pyplot as plt
import meshio
import netCDF4
import numpy as np
import os
import scipy.interpolate
import time

from form_factors import FormFactorMatrix

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


def _main():
    # First, load the DEM of the lunar south pole, which is stored as
    # a netCDF4 file, and pull out the coordinate data.
    path = os.path.join('.', 'tmp128.grd')
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

    h = X[1] - X[0]
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    nx = int(np.round((xmax - xmin)/h))
    ny = int(np.round((ymax - ymin)/h))

    X_mesh, Y_mesh = np.meshgrid(
        np.linspace(xmin, xmax, nx//2),
        np.linspace(ymin, ymax, ny//2)
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

    start_time = time.time()

    # Next, we need to set up Embree. The lines below allocate some
    # memory that Embree manages, and loads our vertices and index
    # lists for the faces. In Embree parlance, we create a "device",
    # which manages a "scene", which has one "geometry" in it, which
    # is our mesh.
    device = embree.Device()
    geometry = device.make_geometry(embree.GeometryType.Triangle)
    scene = device.make_scene()
    vertex_buffer = geometry.set_new_buffer(
        embree.BufferType.Vertex,        # buf_type
        0,                               # slot
        embree.Format.Float3,            # fmt
        3*np.dtype('float32').itemsize,  # byte_stride
        V.shape[0],                      # item_count
    )
    vertex_buffer[:] = V[:]
    index_buffer = geometry.set_new_buffer(
        embree.BufferType.Index,        # buf_type
        0,                              # slot
        embree.Format.Uint3,            # fmt
        3*np.dtype('uint32').itemsize,  # byte_stride,
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
    end_time = time.time()
    print('assembled form factor matrix (%1.2f MB)' %
          (FF.nbytes/(1024**2),))
    print('problem size %d, time elapsed %f, memory %1.2f MB' % (
        len(X), end_time-start_time, FF.nbytes/1024**2))

    # Python makes it very easy to serialize object hierarchies and
    # write them to disk as binary files. We do that here to save the
    # compressed form factor matrix. We can reload it later if we
    # want, without having to first compute it (or load an OBJ file,
    # or set up Embree, or any of that stuff).
    FF.save('FF.bin')
    print('saved compressed form factor matrix to FF.bin')


if __name__ == '__main__':
    _main()