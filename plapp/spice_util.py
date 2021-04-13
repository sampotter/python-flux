# from plasci.config import body, frame
from plapp.config import FluxOpt

def get_sundir(utc0,utc1,stepet=1, V=[]):
    '''This script uses SPICE to compute a trajectory for the sun, loads a
        shape model discretizing a patch of the lunar south pole (made using
        lsp_make_obj.py), and a compressed form factor matrix for that
        shape model (computed using lsp_compress_form_factor_matrix.py).
        It then proceeds to compute the steady state temperature at each sun
        position, writing a plot of the temperature to disk for each sun
        position.

        '''
    import colorcet as cc
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import spiceypy as spice
    import flux.compressed_form_factors as cff
    # from flux.model import get_T
    from flux.plot import tripcolor_vector
    from flux.shape import TrimeshShapeModel

    clktol = '10:000'
    spice.kclear()

    spice.furnsh(f'{FluxOpt.get("example_dir")}/simple.furnsh')
    et0 = spice.str2et(utc0)
    et1 = spice.str2et(utc1)
    nbet = int(np.ceil((et1 - et0) / stepet))
    et = np.linspace(et0, et1, nbet)
    # Sun positions over time period
    possun = spice.spkpos('SUN', et, FluxOpt.get("frame"), 'LT+S', FluxOpt.get("body"))[0]

    if False:
        print(V)
        grdx_ = V[:1,0]
        grdy_ = V[:1,1]
        grdz_ = V[:1,2]

        # Unproject from stereographic to lon/lat
        lon0, lat0, R = 0, -90, 1737.4

        rho = np.sqrt(grdx_ ** 2 + grdy_ ** 2)
        c = 2 * np.arctan2(rho, 2 * R)

        lat_ = np.rad2deg(
            np.arcsin(np.cos(c) * np.sin(np.deg2rad(lat0)) +
                      (np.cos(np.deg2rad(lat0)) * np.sin(c) * grdy_) / rho))

        lon_ = np.mod(
            lon0 + np.rad2deg(
                np.arctan2(
                    grdx_ * np.sin(c),
                    np.cos(np.deg2rad(lat0)) * rho * np.cos(c)
                    - np.sin(np.deg2rad(lat0)) * grdy_ * np.sin(c))),
            360)

        lat_[(grdx_ == 0) & (grdy_ == 0)] = lat0
        lon_[(grdx_ == 0) & (grdy_ == 0)] = lon0

        # Go from lon/lat to cartesian

        az = np.deg2rad(lon_)
        el = np.deg2rad(lat_)
        r = R + grdz_

        x = r * np.cos(az) * np.cos(el)
        y = r * np.sin(az) * np.cos(el)
        z = r * np.sin(el)

        postile = np.vstack([x,y,z]).T
    else:
        postile = 0

    possun = possun+postile

    lonsun = np.arctan2(possun[:, 1], possun[:, 0])
    lonsun = np.mod(lonsun, 2 * np.pi)
    radsun = np.sqrt(np.sum(possun[:, :2] ** 2, axis=1))
    latsun = np.arctan2(possun[:, 2], radsun)
    sun_dirs = np.array([
        np.cos(lonsun) * np.cos(latsun),
        np.sin(lonsun) * np.cos(latsun),
        np.sin(latsun)
    ]).T
    return sun_dirs