#!/usr/bin/env python
import logging
import pickle

import numpy as np

import flux.compressed_form_factors as cff

from flux.form_factors import get_form_factor_matrix
from flux.shape import EmbreeTrimeshShapeModel, CgalTrimeshShapeModel

def setup_form_factor_matrix(compress=True, tol=1e-2, min_size=256, engine='cgal'):
    """
    Loads facets and produces FF
    Args:
        tol: max error for SVD truncation?
        compress: bool, get full or compressed FF
    """
    # Load mesh
    V = np.load('lsp_V.npy')
    F = np.load('lsp_F.npy')
    N = np.load('lsp_N.npy')

    # Set up shape model and build form factor matrix
    if engine == 'cgal':
        shape_model = CgalTrimeshShapeModel(V.copy(order='C'), F.copy(order='C'), N.copy(order='C'))
    elif engine == 'embree':
        shape_model = EmbreeTrimeshShapeModel(V.copy(order='C'), F.copy(order='C'), N.copy(order='C'))
    else:
        logging.error("Please specify which ray tracing engine to use: cgal or embree.")

    if compress:
        FF = cff.CompressedFormFactorMatrix(
            shape_model,
            tol=tol,
            min_size=min_size,
            RootBlock=cff.FormFactorQuadtreeBlock)

        print(f'- assembled FF [depth={FF.depth}]')

        FF.save('lsp_compressed_form_factors.bin')
        print('- wrote FF matrix to lsp_compressed_form_factors.bin')
    else:
        FF = get_form_factor_matrix(shape_model)
        with open('lsp_full_form_factors.bin', 'wb') as f:
            pickle.dump(FF,f)

if __name__ == '__main__':

    setup_form_factor_matrix()
