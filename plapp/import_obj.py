import meshio
import numpy as np
from flux.shape import get_centroids, get_surface_normals

def import_obj(obj_path, get_normals=False):
    # use meshio to import obj shapefile
    mesh = meshio.read(
        filename=obj_path,  # string, os.PathLike, or a buffer/open file
        file_format="obj",  # optional if filename is a path; inferred from extension
    )
    # provides
    # mesh.points = V
    # mesh.cells = F - 1 (subtracted from all components, not sure why but adding it again generates warnings)
    V = mesh.points
    V = V.astype(np.float32)  # embree is anyway single precision
    F = mesh.cells[0].data

    if get_normals:
        P = get_centroids(V, F)
        N = get_surface_normals(V, F)
        N[(N * P).sum(1) < 0] *= -1
        return V, F, N
    else:
        return V, F