#!/usr/bin/env python

DO_3D_PLOTTING = False

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
    import dmsh
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
import pyvista as pv
import scipy.interpolate
import time

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.form_factors import get_form_factor_matrix
from flux.model import compute_steady_state_temp
from flux.plot import plot_blocks, tripcolor_vector
from flux.shape import CgalTrimeshShapeModel

if __name__ == '__main__':
    # Define constants used in the simulation:
    e0 = 3*np.pi/180 # Solar elevation angle
    F0 = 1365 # Solar constant
    emiss = 0.95 # Emissitivity
    rho = 0.12 # Visual (?) albedo
    dir_sun = np.array([0, -np.cos(e0), np.sin(e0)]) # Direction of sun
    x0, x1, y0, y1 = -35, 25, 40, 100 # Bounding box of crater (not
                                      # used except in
                                      # lunar_south_pole.png plot at
                                      # the moment)
    h = 0.25 # Desired edge lengths of triangles generated by DistMesh
    p0 = np.array([-5, 75]) # Center of circular mesh
    r0 = 25 # # Radius of circular mesh

    # First, load the DEM of the lunar south pole, which is stored as
    # a netCDF4 file, and pull out the coordinate data.
    path = os.path.join('.', 'lunar_south_pole_80mpp_curvature.grd')
    rootgrp = netCDF4.Dataset(path)
    X = np.array(rootgrp.variables['x'])
    Y = np.array(rootgrp.variables['y'])
    Z = np.array(rootgrp.variables['z'])
    del rootgrp
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
        # Here, we use dmsh to mesh Haworth. We use the library dmsh
        # which reimplements distmesh in Python. This snippet of code
        # could modified without too much trouble to mesh the entire
        # rectangular bounding box, and to vary the target edge length
        # ("fh" below).
        print('meshing crater by using distmesh')
        geo = dmsh.Circle(p0, r0)
        V, F = dmsh.generate(geo, h)
    else:
        print('meshing crater by making Delaunay triangulation on grid')
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

    # Since Embree runs in single precision, there's no reason to use
    # double precision here.
    V = V.astype(np.float32)

    # Create a triangle mesh shape model using the vertices (V) and
    # face indices (F).
    shape_model = CgalTrimeshShapeModel(V, F)

    # Build the compressed form factor matrix. All of the code related
    # to this can be found in the "form_factors.py" file in this
    # directory.
    t0 = time.perf_counter()
    FF = CompressedFormFactorMatrix(shape_model, tol=1e-3, min_size=512)
    print('assembled form factor matrix in %f sec (%1.2f MB)' %
          (time.perf_counter() - t0, FF.nbytes/(1024**2),))
    del t0

    # Python makes it very easy to serialize object hierarchies and
    # write them to disk as binary files. We do that here to save the
    # compressed form factor matrix. We can reload it later if we
    # want, without having to first compute it (or load an OBJ file,
    # or set up Embree, or any of that stuff).
    FF.save('FF.bin')
    print('saved compressed form factor matrix to FF.bin')

    # Compute the direct irradiance and find the elements which are
    # in shadow.
    E = shape_model.get_direct_irradiance(F0, dir_sun, unit_Svec=True)
    I_shadow = E == 0

    # Compute the steady state temperature. This function is at the
    # top of this file.
    T = compute_steady_state_temp(FF, E, rho, emiss)
    print('computed T')

    # Finally, we make some plots showing what we just did, and write
    # them to disk:

    fig, ax = plot_blocks(FF._root)
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

    # Use pyvista to make a nice 3D plot of the temperature.
    vertices = V.copy()
    faces = np.concatenate([3*np.ones((F.shape[0], 1), dtype=F.dtype), F], axis=1)
    surf = pv.PolyData(vertices, faces)
    surf.cell_arrays['T'] = T
    surf.cell_arrays['opacity'] = np.logical_not(I_shadow).astype(T.dtype)

    this_cmap = cmap['jet']

    if DO_3D_PLOTTING:
        plotter = pv.Plotter()
        plotter.add_mesh(surf, scalars='T', opacity='opacity',
                         use_transparency=True, cmap=this_cmap)
        cpos = plotter.show()

        plotter = pv.Plotter(off_screen=True)
        plotter.background_color = 'black'
        plotter.add_mesh(surf, scalars='T', opacity='opacity',
                         use_transparency=True, cmap=this_cmap)
        plotter.camera_position = cpos
        plotter.set_focus([*p0, P[:, 2].mean()])
        plotter.screenshot('test.png')
