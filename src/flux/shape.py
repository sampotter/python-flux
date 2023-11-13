try:
    import embree
except:
    pass
import numpy as np
import sys

from abc import ABC
try:
    from flux.cgal.aabb import AABB
except:
    pass

use1M = True

def get_centroids(V, F):
    return V[F].mean(axis=1)


def get_cross_products(V, F):
    V0 = V[F][:, 0, :]
    C = np.cross(V[F][:, 1, :] - V0, V[F][:, 2, :] - V0)
    return C


def get_face_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    A = C_norms/2
    return A


def get_surface_normals(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    return N


def get_surface_normals_and_face_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    A = C_norms/2
    return N, A


class ShapeModel(ABC):
    pass


class TrimeshShapeModel(ShapeModel):
    """A shape model consisting of a single triangle mesh."""

    def __init__(self, V, F, N=None, P=None, A=None):
        """Initialize a triangle mesh shape model. No assumption is made about
        the way vertices or faces are stored when building the shape
        model except that V[F] yields the faces of the mesh. Vertices
        may be repeated or not.

        Parameters
        ----------
        V : array_like
            An array with shape (num_verts, 3) whose rows correspond to the
            vertices of the triangle mesh
        F : array_like
            An array with shape (num_faces, 3) whose rows index the faces
            of the triangle mesh (i.e., V[F] returns an array with shape
            (num_faces, 3, 3) such that V[F][i] is a 3x3 matrix whose rows
            are the vertices of the ith face.
        N : array_like, optional
            An array with shape (num_faces, 3) consisting of the triangle
            mesh face normals. Can be passed to specify the face normals.
            Otherwise, the face normals will be computed from the cross products
            of the face edges (i.e. np.cross(vi1 - vi0, vi2 - vi0) normalized).
        P : array_like, optional
            An array with shape (num_faces, 3) consisting of the triangle
            centroids. Can be optionally passed to avoid recomputing.
        A : array_like, optional
            An array of shape (num_faces,) containing the triangle areas. Can
            be optionally passed to avoid recomputing.

        """
        if type(self) == TrimeshShapeModel:
            raise RuntimeError("tried to instantiate TrimeshShapeModel directly")

        self.dtype = V.dtype

        self.V = V
        self.F = F

        if N is None and A is None:
            N, A = get_surface_normals_and_face_areas(V, F)
        elif A is None:
            if N.shape[0] != F.shape[0]:
                raise Exception(
                    'must pass same number of surface normals as faces (got ' +
                    '%d faces and %d normals' % (F.shape[0], N.shape[0])
                )
            A = get_face_areas(V, F)
        elif N is None:
            N = get_surface_normals(V, F)

        self.P = get_centroids(V, F)
        self.N = N
        self.A = A

        assert self.P.dtype == self.dtype
        assert self.N.dtype == self.dtype
        assert self.A.dtype == self.dtype

        self._make_scene()

    def __reduce__(self):
        return (self.__class__, (self.V, self.F, self.N, self.P, self.A))

    def __repr__(self):
        return 'a TrimeshShapeModel with %d vertices and %d faces' % (
            self.num_verts, self.num_faces)

    @property
    def num_faces(self):
        return self.P.shape[0]

    @property
    def num_verts(self):
        return self.V.shape[0]

    def intersect1(self, x, d):
        '''Trace a single ray starting from `x` and in the direction `d`.  If
        there is a hit, return the index (`i`) of the hit and a
        parameter `t` such that the hit point is given by `x(t) = x +
        t*d`.

        '''
        return self._intersect1(x, d)

    def get_visibility(self, I, J, oriented=False):

        '''Compute the visibility mask for pairs of indices (i, j) taken from
        index arrays I and J. If m = len(I) and N = len(J), the
        resulting array is an m x N binary matrix V, where V[i, j] ==
        1 if a ray traced from the centroid of facet i to the centroid
        of facet j is unoccluded.

        If oriented is True, then this will use the surface normal to
        check whether both triangles have the correct orientation (the
        normals and the vector pointing from the centroid of one
        triangle to the other have a positive dot product---i.e., the
        triangles are facing one another).

        '''
        vis = self._get_visibility(I, J)

        # set vis for any pairs of faces with unoccluded LOS which
        # aren't oriented towards each other to False
        if oriented:
            I_, J_ = np.where(vis)
            mask = ((self.P[J_] - self.P[I_])*self.N[I_]).sum(1) <= 0
            for i, j in zip(I_[mask], J_[mask]):
                vis[i, j] = False

        return vis

    def get_visibility_1_to_N(self, i, J, oriented=False):
        '''Convenience function for calling get_visibility with a single
        source triangle.

        '''
        return self.get_visibility([i], J, oriented).ravel()

    def get_visibility_matrix(self, oriented=False):
        '''Convenience function for computing the visibility matrix. This just
        calls get_visibility(I, I, eps, oriented), where I =
        np.arange(num_faces).

        '''
        I = np.arange(self.num_faces, dtype=np.uintp)
        return self.get_visibility(I, I, oriented)

    def is_occluded(self, I, D):
        '''Check whether a ray shot from the centroid of the triangles indexed
        by I is occluded by any other part of the mesh, where the ray
        direction is specified by D. If D is a single vector, then
        each face will use the same ray direction.

        '''
        return self._is_occluded(I, D)

    def get_direct_irradiance(self, F0, Dsun, basemesh=None, eps=None):
        '''Compute the insolation from the sun.

        Parameters
        ----------
        F0: float
            The solar constant. [W/m^2]

        Dsun: numpy.ndarray
            A length 3 vector or Mx3 array of sun directions: vectors
            indicating the direction of the sun in world coordinates.

        basemesh: same as self, optional
            mesh used to check (Sun, light source) visibility at "self.cells";
            it would usually cover a larger area than "self".

        Returns
        -------
        E: numpy.ndarray
            A vector of length self.num_faces or an array of size
            M x self.num_faces, where M is the number of sun
            directions.

        '''
        # Determine which rays escaped (i.e., can see the sun)
        if basemesh is None:
            I = ~self.is_occluded(np.arange(self.num_faces), Dsun)
        else:
            I = ~basemesh.is_occluded(np.arange(self.num_faces), Dsun)

        # Compute the direct irradiance
        if Dsun.ndim == 1:
            E = np.zeros(self.num_faces, dtype=self.dtype)
            E[I] = F0*np.maximum(0, self.N[I]@Dsun)
        elif (Dsun.ndim == 2) & (Dsun.shape[0] == self.num_faces): # TODO check if useful and if it still works
            if Dsun.shape[0] != self.num_faces:
                raise ValueError('need Dsun.shape[0] == num_faces')
            if Dsun.shape[1] != 3:
                raise ValueError('need Dsun.shape[1] == 3 if Dsun.ndim == 2')
            E = np.zeros(self.num_faces, dtype=self.dtype)
            E[I] = F0*np.maximum(0, (self.N[I]*Dsun[I]).sum(1))
        elif Dsun.ndim == 2:  # for discretized extended light source
            if Dsun.shape[1] != 3:
                raise ValueError('need Dsun.shape[1] == 3 if Dsun.ndim == 2')
            E = self.N@Dsun.T
            # E = np.einsum(self.N,[0,1],Dsun,[2,1]) # same as '@', easier to generalize if needed
            E = np.where(I, E, 0)
            E = np.mean(F0)*np.maximum(0, np.sum(E,axis=1)/Dsun.shape[0]) # F0 doesn't varies by <1 W/m2 within discretized source
        else:
            raise RuntimeError('Dsun.ndim > 2 not implemented yet')

        return E


    def get_pyvista_unstructured_grid(self):
        try:
            import pyvista as pv
        except:
            raise ImportError('failed to import PyVista')

        try:
            import vtk as vtk
        except:
            raise ImportError('failed to import vtk')

        return pv.UnstructuredGrid({vtk.VTK_TRIANGLE: self.F}, self.V)


class CgalTrimeshShapeModel(TrimeshShapeModel):
    def _make_scene(self):
        if 'flux.cgal.aabb' not in sys.modules:
            raise ImportError('failed to import AABB from flux.cgal.aabb')

        self.aabb = AABB.from_trimesh(
            self.V.astype(np.float64), self.F.astype(np.uintp))

    def _intersect1(self, x, d):
        return self.aabb.intersect1(x, d)

    def _get_visibility(self, I, J):
        if not isinstance(I, np.ndarray):
            I = np.array(I)
        I = I.astype(np.uintp)

        if not isinstance(J, np.ndarray):
            J = np.array(J)
        J = J.astype(np.uintp)

        return self.aabb.test_face_to_face_vis_MN(I, J)

    def _is_occluded(self, I, D):
        m = len(I)
        if D.ndim == 1:
            occluded = np.empty(m, dtype=np.bool_)
            for p, i in enumerate(I):
                occluded[p] = self.aabb.ray_from_centroid_is_occluded(i, D)
        elif D.ndim == 2:
            occluded = np.empty((m,D.shape[0]), dtype=np.bool_)
            occluded[:,:] = self.aabb.ray_from_centroid_is_occluded_2d(I, D)
        return occluded


class EmbreeTrimeshShapeModel(TrimeshShapeModel):
    def _make_scene(self):
        '''Set up an Embree scene. This function allocates some memory that
        Embree manages, and loads vertices and index lists for the
        faces. In Embree parlance, this function creates a "device",
        which manages a "scene", which has one "geometry" in it, which
        is our mesh.

        '''
        if 'embree' not in sys.modules:
            raise ImportError('failed to import embree')

        device = embree.Device()

        geometry = device.make_geometry(embree.GeometryType.Triangle)
        # geometry.set_build_quality(embree.BuildQuality.High)

        scene = device.make_scene()
        # scene.set_build_quality(embree.BuildQuality.High)
        scene.set_flags(embree.SceneFlags.Robust)

        vertex_buffer = geometry.set_new_buffer(
            embree.BufferType.Vertex, # buf_type
            0, # slot
            embree.Format.Float3, # fmt
            3*np.dtype('float32').itemsize, # byte_stride
            self.V.shape[0], # item_count
        )
        vertex_buffer[:] = self.V[:]

        index_buffer = geometry.set_new_buffer(
            embree.BufferType.Index, # buf_type
            0, # slot
            embree.Format.Uint3, # fmt
            3*np.dtype('uint32').itemsize, # byte_stride,
            self.F.shape[0]
        )
        index_buffer[:] = self.F[:]

        geometry.commit()

        scene.attach_geometry(geometry)

        geometry.release()

        scene.commit()

        # This is the only variable we need to retain a reference to
        # (I think)
        self.scene = scene

    def _intersect1(self, x, d):
        raise RuntimeError('intersect1 no implemented for EmbreeTrimeshShapeModel')

    def _get_visibility(self, I, J):
        # TODO: desperately need a better way to set this.
        eps = 1e3*np.finfo(np.float32).resolution

        m, n = len(I), len(J)

        PJ = self.P[J]

        D = np.empty((m*n, 3), dtype=self.dtype)
        for q, i in enumerate(I):
            D[q*n:(q + 1)*n] = PJ - self.P[i]
        D_norm_sq = np.sqrt(np.sum(D**2, axis=1))
        mask = D_norm_sq > eps
        D = D[mask]/D_norm_sq[mask].reshape(-1, 1)

        P = np.empty((m*n, 3), dtype=self.dtype)
        for q, i in enumerate(I):
            P[q*n:(q + 1)*n] = self.P[i]

        J_extended = np.empty((m*n,), dtype=J.dtype)
        for q in range(m):
            J_extended[q*n:(q + 1)*n] = J
        J_extended = J_extended[mask]

        num_masked = mask.sum()

        rayhit = embree.RayHit1M(num_masked)

        context = embree.IntersectContext()
        context.flags = embree.IntersectContextFlags.COHERENT

        rayhit.org[:] = P[mask] + eps*D
        rayhit.dir[:] = D
        rayhit.tnear[:] = 0
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

        if use1M:
            self.scene.intersect1M(context, rayhit)
        else:
            self.scene.intersectNp(context, rayhit)

        vis = np.ones((m*n,), dtype=bool) # vis by default
        vis[mask] = np.logical_and(
            rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
            rayhit.prim_id == J_extended
        )

        return vis.reshape(m, n)

    def _is_occluded(self, I, D):
        if D.ndim != 1 and D.ndim != 2:
            raise ValueError('D.ndim should be 1 or 2')

        # TODO: see comment in _get_visibility
        eps = 1e3*np.finfo(np.float32).resolution

        m = len(I)

        ray = embree.Ray1M(m)
        ray.org[:] = self.P[I] + eps*self.N[I]
        ray.dir[:] = D
        ray.tnear[:] = 0
        ray.tfar[:] = np.inf
        ray.flags[:] = 0

        context = embree.IntersectContext()
        context.flags = embree.IntersectContextFlags.COHERENT

        self.scene.occluded1M(context, ray)

        return np.logical_not(np.isposinf(ray.tfar))


trimesh_shape_models = [
    CgalTrimeshShapeModel,
    EmbreeTrimeshShapeModel
]
