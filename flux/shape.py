import embree
import numpy as np


from abc import ABC

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

    def _make_scene(self):
        '''Set up an Embree scene. This function allocates some memory that
        Embree manages, and loads vertices and index lists for the
        faces. In Embree parlance, this function creates a "device",
        which manages a "scene", which has one "geometry" in it, which
        is our mesh.

        '''
        device = embree.Device()
        geometry = device.make_geometry(embree.GeometryType.Triangle)
        scene = device.make_scene()
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

    def __reduce__(self):
        return (self.__class__, (self.V, self.F, self.N, self.P, self.A))

    def __repr__(self):
        return 'a TrimeshShapeModel with %d vertices and %d faces' % (
            self.num_verts, self.num_faces)

    @property
    def num_faces(self):
        return self.P.shape[0]

    def check_vis(self, I, J, eps=None):
    @property
    def num_verts(self):
        return self.V.shape[0]

        '''Compute the visibility mask for pairs of indices (i, j) taken from
        index arrays I and J. If M = len(I) and N = len(J), the
        resulting array is an M x N binary matrix V, where V[i, j] ==
        1 if a ray traced from the centroid of facet i to the centroid
        of facet j is unoccluded.

        The parameter eps is used to perturb the start of each ray
        away from the facet centroid. This is because Embree (by
        default) doesn't know to check for self-intersection. A
        "filter function" should be set up to support this, but this
        hasn't been implemented. For now, we use the eps parameter,
        which is a bit of a hack.

        '''
        if eps is None:
            eps = 1e3*np.finfo(np.float32).resolution
            # TODO clean up how the "shift along N" is defined
            # also applied along P
            # (currently proportional to facet side)
            eps = np.sqrt(self.A[i]) / 200

        M, N = len(I), len(J)

        PJ = self.P[J]

        D = np.empty((M*N, 3), dtype=self.dtype)
        for q, i in enumerate(I):
            D[q*N:(q + 1)*N] = PJ - self.P[i]
        D /= np.sqrt(np.sum(D**2, axis=1)).reshape(D.shape[0], 1)

        P = np.empty_like(D)
        for q, i in enumerate(I):
            P[q*N:(q + 1)*N] = self.P[i]
        P += eps*D

        rayhit = embree.RayHit1M(M*N)

        context = embree.IntersectContext()
        # context.flags = embree.IntersectContextFlags.COHERENT

        rayhit.org[:] = P + eps*self.N[i]
        rayhit.dir[:] = D
        rayhit.tnear[:] = 0
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

        if use1M:
            self.scene.intersect1M(context, rayhit)
        else:
            self.scene.intersectNp(context, rayhit)


        return np.logical_and(
            rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
            rayhit.prim_id == J
        ).reshape(M, N)

    def check_vis_1_to_N(self, i, J, eps=None):
        return self.check_vis([i], J, eps).ravel()

    def get_direct_irradiance(self, F0, Dsun, unit_Svec=False, basemesh=None, eps=None):
        '''Compute the insolation from the sun.

        Parameters
        ----------
        F0: float
            The solar constant. [W/m^2]

        Dsun: numpy.ndarray
            An length 3 vector or Mx3 array of sun directions: vectors
            indicating the direction of the sun in world coordinates.

        basemesh: same as self, optional
            mesh used to check (Sun, light source) visibility at "self.cells";
            it would usually cover a larger area than "self".

        eps: float
            How far to perturb the start of the ray away from each
            face. Default is 1e3*np.finfo(np.float32).resolution. This
            is to overcome precision issues with Embree.

        unit_Svec: bool
            defines if Dsun is a unit vector (Sun direction) or
            the actual Sun-origin vector (check AU units below)

        Returns
        -------
        E: numpy.ndarray
            A vector of length self.num_faces or an array of size
            M x self.num_faces, where M is the number of sun
            directions.

        '''
        if eps is None:
            eps = 1e3*np.finfo(np.float32).resolution

        if basemesh == None:
            basemesh = self

        # Here, we use Embree directly to find the indices of triangles
        # which are directly illuminated (I_sun) or not (I_shadow).

        n = self.num_faces

        if Dsun.ndim == 1:
            # Normalize Dsun
            distSunkm = np.sqrt(np.sum(Dsun ** 2))
            # print(distSunkm)
            Dsun /= distSunkm

            ray = embree.Ray1M(n)
            if eps.ndim==0:
                ray.org[:] = self.P + eps*self.N
            else:
                ray.org[:] = self.P + eps[:,np.newaxis]*self.N
            ray.dir[:] = Dsun
            ray.tnear[:] = 0
            ray.tfar[:] = np.inf
            ray.flags[:] = 0
        elif Dsun.ndim == 2:
            # Normalize Dsun
            distSunkm = np.linalg.norm(Dsun,axis=1)[:,np.newaxis]
            Dsun /= distSunkm

            m = Dsun.size//3
            ray = embree.Ray1M(m*n)
            for i in range(m):
                ray.org[i*n:(i + 1)*n, :] = self.P + eps[:,np.newaxis]*self.N
            for i, d in enumerate(Dsun):
                ray.dir[i*n:(i + 1)*n, :] = d
            ray.tnear[:] = 0
            ray.tfar[:] = np.inf
            ray.flags[:] = 0

        context = embree.IntersectContext()
        basemesh.scene.occluded1M(context, ray)
        # Determine which rays escaped (i.e., can see the sun)
        I = np.isposinf(ray.tfar)

        # rescale solar flux depending on distance
        if not unit_Svec:
            AU_km = 149597900.
            F0 *= (AU_km / distSunkm) ** 2

        # Compute the direct irradiance
        if Dsun.ndim == 1:
            E = np.zeros(n, dtype=self.dtype)
            E[I] = F0*np.maximum(0, self.N[I]@Dsun)
        else:
            E = np.zeros((n, m), dtype=self.dtype)
            I = I.reshape(m, n)
            # TODO check if this can be vectorized
            for i, d in enumerate(Dsun):
                if unit_Svec:
                    E[I[i], i] = F0*np.maximum(0, self.N[I[i]]@d)
                else:
                    E[I[i], i] = F0[i]*np.maximum(0, self.N[I[i]]@d)
        return E
