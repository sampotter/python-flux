#!/usr/bin/env python3
# ----------------------------------
# Read shape from obj file and join to py-flux pipeline
# ----------------------------------
# Author: Stefano Bertone
# Created: 17-Nov-2020
#
import logging
import pickle
import os

from tqdm import tqdm

from plapp.config import FluxOpt
from plapp.grd2obj import grd2obj
from plapp.import_obj import import_obj
from plapp.spice_util import get_sundir
from flux.model import compute_steady_state_temp

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


import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import time

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.plot import plot_blocks, tripcolor_vector
from flux.shape import TrimeshShapeModel

# Define constants used in the simulation:
F0 = 1365  # Solar constant
emiss = 0.95  # Emissitivity
rho = 0.12  # Visual (?) albedo

def main(filein):

    start = time.time()

    # create useful dirs
    basedir = os.path.join(FluxOpt.get("example_dir"), FluxOpt.get("body"))+'/'
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    # Generate obj from topography map of selected region (at selected body) -- if planet
    if FluxOpt.get('body') == 'MOON' and FluxOpt.get("new_FF") and FluxOpt.get("new_obj"):
        objin = f'{basedir}{FluxOpt.get("feature")}.obj'
        # if not os.path.exists(filein):
        input_grd = filein
        mesh_path = grd2obj(grdfile=os.path.join(FluxOpt.get("example_dir"), input_grd),
                                name=FluxOpt.get("feature"),
                                crater=FluxOpt.get("is_crater"),
                                h=FluxOpt.get("resolution")) # km
        print(f"Tri-meshes for {input_grd} generated and saved to {mesh_path}.")
    # else just organize input obj
    else:
        # link shape obj to dir
        objlink = basedir+FluxOpt.get("body")+'.obj'
        # print(objlink)
        if os.path.islink(objlink):
            os.remove(objlink)
        os.symlink(filein, objlink)
        filein = objlink

    # use meshio to extract vertices and facets from obj file for inner and, if requested, global mesh
    filein = f'{basedir}{FluxOpt.get("feature")}'
    print(f"Reading mesh from {filein}...")
    V, F, N = import_obj(obj_path=f"{filein}.obj", get_normals=True)
    if FluxOpt.get("use_distant_topo"):
        V_ext,F_ext, N_ext = import_obj(obj_path=f"{filein}_ext" + ".obj", get_normals=True)

    # Generate FF
    # -----------
    # Create a triangle mesh shape model using the vertices (V) and
    # face indices (F).
    shape_model = TrimeshShapeModel(V, F, N)

    # Build the compressed form factor matrix. All of the code related
    # to this can be found in the "form_factors.py" file in this
    # directory.
    # if it already exists, and not re-created, just read it
    binout = basedir +'FF_' + FluxOpt.get('body') + '_' + FluxOpt.get("tree_kind") + '.bin'
    if FluxOpt.get("new_FF"):

        print("Started assembling form factor matrix ...")
        t0 = time.perf_counter()
        FF = CompressedFormFactorMatrix.assemble(
            shape_model, tree_kind=FluxOpt.get("tree_kind"), tol=1e-7)
        print('assembled form factor matrix in %f sec (%1.2f MB)' %
              (time.perf_counter() - t0, FF.nbytes / (1024 ** 2),))
        del t0

        # Python makes it very easy to serialize object hierarchies and
        # write them to disk as binary files. We do that here to save the
        # compressed form factor matrix. We can reload it later if we
        # want, without having to first compute it (or load an OBJ file,
        # or set up Embree, or any of that stuff).
        FF.save(binout)
        print('saved compressed form factor matrix to FF_' + FluxOpt.get('body') + '.bin')

    else:
        FF = pickle.load(open(binout, "rb"))
        print('Compressed form factor matrix loaded from FF_' + FluxOpt.get('body') + '.bin.')
        # print(FF.nbytes / (1024 ** 2))
        print('Characteristics:\n size %1.2f MB' % (FF.nbytes / (1024 ** 2)))

    print("Size:",FF.shape)
    print("Compression ratio:",(FF.nbytes) / (FF.shape[0]**2 * 8) * 100,"%")

    V1 = V.copy()
    V1[:, 0] = V[:, 1]
    V1[:, 1] = V[:, 0]
    V = V1.copy()
    fig, ax = tripcolor_vector(V, F, V[:,2]+1737.4, cmap=cmap['jet'])

    if FluxOpt.get("debug"):
        fig, ax = plot_blocks(FF._root)
        fig.savefig(basedir + FluxOpt.get('body') + '_' + FluxOpt.get("tree_kind") + '_blocks.png')
        plt.close(fig)
        print('Wrote ' + FluxOpt.get('body') + '_' + FluxOpt.get("tree_kind") + '_blocks.png to disk.')

    # Get Sun direction w.r.t. body/region of interest
    # using spice (kernels need to be loaded using mymeta file).
    # A default elevation is used if use_spice is False --> 1 element array
    ## sun_dirs is an array of normalized directions Sun-body at requested times
    # --------------------------------------------------
    if FluxOpt.get("use_spice"):
        # Define time window
        utc0 = FluxOpt.get("spice_utc_start") #'2011 MAR 01 00:00:00.00'
        utc1 = FluxOpt.get("spice_utc_end") #'2011 MAR 02 00:00:00.00'
        stepet = FluxOpt.get("spice_utc_stepet") # 3*3600

        sun_dirs = get_sundir(utc0=utc0, utc1=utc1, stepet=stepet, V=V)
    else:
        e0 = 3 * np.pi / 180  # Solar elevation angle
        sun_dirs = np.array([[0, -np.cos(e0), np.sin(e0)]])  # Direction of sun

    E_arr = []
    # Compute steady state temperature
    # for all Sun positions
    # --------------------------------
    logging.info(msg="Get direct irradiance steps...")
    start_from_frame = 0
    for idx, sun_dir in tqdm(enumerate(sun_dirs[start_from_frame:]),total=len(sun_dirs[start_from_frame:])):
        # print('frame = %d' % i, end='')
        idx += start_from_frame
        # Compute the direct irradiance and find the elements which are
        # in shadow.
        # If limited region, the option to use surrounding topography
        # for ray-tracing and occultations is given (option: use_distant_topo)
        if FluxOpt.get("is_crater") and FluxOpt.get("use_distant_topo"):
            # Create a triangle mesh shape model using the vertices (V) and
            # face indices (F).
            shape_model_ext = TrimeshShapeModel(V_ext, F_ext, N_ext)
            # print(f"Sun (r,lat,lon): {np.rad2deg(cart2sph(sun_dir)[1:])}")
            E = shape_model.get_direct_irradiance(F0, sun_dir,basemesh=shape_model_ext)
        else:
            E = shape_model.get_direct_irradiance(F0, sun_dir)
        I_shadow = E == 0

        E_arr.append(E)

    # Stack time-steps to E columns
    E = np.vstack(E_arr).T
    logging.info(msg="Plot E for all sun_dirs...")
    for idx, sun_dir in tqdm(enumerate(sun_dirs), total=len(sun_dirs)):

        # Finally, we make some plots showing what we just did, and write
        # them to disk:
        fig, ax = tripcolor_vector(V, F, E[:,idx], vmax=700, cmap=cmap['jet'])
        fig.savefig(basedir + FluxOpt.get('body') + '_' + FluxOpt.get("tree_kind") + '_E_%03d.png' % idx)
        plt.close(fig)

    # compute T
    T = compute_steady_state_temp(FF, E, rho, emiss)
    T = np.vstack(T).T

    logging.info(msg="Plot T,F for all sun_dirs...")
    for idx, sun_dir in tqdm(enumerate(sun_dirs), total=len(sun_dirs)):

        # Finally, we make some plots showing what we just did, and write
        # them to disk:
        fig, ax = tripcolor_vector(V, F, T[:,idx], vmin=0, vmax=400, cmap=cmap['fire'])
        fig.savefig(basedir + FluxOpt.get('body') + '_' + FluxOpt.get("tree_kind") + '_T_%03d.png' % idx)
        plt.close(fig)
        # print('wrote '+FluxOpt.config('body')+'_'+FluxOpt.config("tree_kind")+'_T.png to disk')

        fig, ax = tripcolor_vector(V, F, T[:,idx], I=I_shadow, cmap=cmap['jet'],vmax=100)
        fig.savefig(basedir + FluxOpt.get('body') + '_' + FluxOpt.get("tree_kind") + '_T_shadow_%03d.png' % idx)
        plt.close(fig)
        # print('wrote '+FluxOpt.config('body')+'_'+FluxOpt.config("tree_kind")+'_T_shadow.png to disk')

        if FluxOpt.get("DO_3D_PLOTTING") and idx == 0:
            # Use pyvista to make a nice 3D plot of the temperature.
            vertices = V.copy()
            faces = np.concatenate([3 * np.ones((F.shape[0], 1), dtype=F.dtype), F], axis=1)
            surf = pv.PolyData(vertices, faces)
            surf.cell_arrays['T'] = T
            surf.cell_arrays['opacity'] = 0.4  # np.logical_not(I_shadow).astype(T.dtype)

            this_cmap = cmap['jet']

            plotter = pv.Plotter()
            plotter.add_mesh(surf, scalars='T', opacity='opacity',
                             use_transparency=True, cmap=this_cmap)
            cpos = plotter.show()

            plotter = pv.Plotter(off_screen=True)
            plotter.background_color = 'black'
            plotter.add_mesh(surf, scalars='T', opacity='opacity',
                             use_transparency=True, cmap=this_cmap)
            plotter.camera_position = cpos
            # plotter.set_focus([*p0, P[:, 2].mean()])
            plotter.screenshot('test.png')

    # real processing and plotting stops here...
    end = time.time()
    print("Ended successfully after", end-start,"seconds!")

if __name__ == '__main__':

    FluxOpt.set('body', 'MOON')
    FluxOpt.set('new_obj', True)
    FluxOpt.set('new_FF', True)
    FluxOpt.set('use_spice', True)
    FluxOpt.set("spice_utc_start", '2021 FEB 15 00:00:00.00')
    FluxOpt.set("spice_utc_end", '2021 MAR 15 00:00:00.00')
    FluxOpt.set("spice_utc_stepet", 24 * 3600)
    FluxOpt.set("use_distant_topo", False)
    FluxOpt.set("tree_kind", "oct")

    # Always append after changing options, to check if options make sense
    FluxOpt.check_consistency()

    # set obj file to import
    if FluxOpt.get('body') == 'MOON':
        flist = ['MOON']

    print('flist:',flist)
    for filein in flist:
        print("Processing", FluxOpt.get('body'), " (obj file from", filein, ") ...")
        main(filein=filein)