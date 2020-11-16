'''This script uses SPICE to compute a trajectory for the sun, loads a
shape model discretizing a patch of the lunar south pole (made using
haworth_make_obj.py), and a compressed form factor matrix for that
shape model (computed using haworth_compress_form_factor_matrix.py).
It then proceeds to compute the steady state temperature at each sun
position, writing a plot of the temperature to disk for each sun
position.

'''

import colorcet as cc
import compressed_form_factors as cff
import matplotlib.pyplot as plt
import numpy as np
import pickle
import spiceypy as spice


from model import get_T
from plot import tripcolor_vector
from shape import TrimeshShapeModel

clktol = '10:000'

spice.kclear()
spice.furnsh('simple.furnsh')

# Define time window

utc0 = '2011 MAR 01 00:00:00.00'
utc1 = '2012 MAR 01 00:00:00.00'
stepet = 3600

et0 = spice.str2et(utc0)
et1 = spice.str2et(utc1)
nbet = int(np.ceil((et1 - et0)/stepet))
et = np.linspace(et0, et1, nbet)

# Sun positions over time period

possun = spice.spkpos('SUN', et, 'MOON_ME', 'LT+S', 'MOON')[0]
lonsun = np.arctan2(possun[:, 1], possun[:, 0])
lonsun = np.mod(lonsun, 2*np.pi)
radsun = np.sqrt(np.sum(possun[:, :2]**2, axis=1))
latsun = np.arctan2(possun[:, 2], radsun)

sun_dirs = np.array([
    np.cos(lonsun)*np.cos(latsun),
    np.sin(lonsun)*np.cos(latsun),
    np.sin(latsun)
]).T

# Use these temporary parameters...

F0 = 1365 # Solar constant
emiss = 0.95 # Emissitivity
rho = 0.12 # Visual (?) albedo

# Load shape model

V = np.load('haworth_V.npy')
F = np.load('haworth_F.npy')
N = np.load('haworth_N.npy')

shape_model = TrimeshShapeModel(V, F, N)

# Load compressed form factor matrix from disk

FF_path = 'haworth_compressed_form_factors.bin'
FF = cff.CompressedFormFactorMatrix.from_file(FF_path)

# Compute steady state temperature

for i in range(100):
    print('frame = %d' % i, end='')
    sun_dir = sun_dirs[i]
    E = shape_model.get_direct_irradiance(F0, sun_dir)
    T, nmul = get_T(FF, E, rho, emiss)
    print(' (nmul = %d)' % nmul)
    fig, ax = tripcolor_vector(V, F, T, cmap=cc.cm.rainbow)
    fig.savefig('haworth_T_%03d.png' % i)
    plt.close(fig)