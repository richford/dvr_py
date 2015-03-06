"""
Use a simple Discrete Variable Representation method to solve
one-dimensional potentials.

A good general introduction to DVR methods is
Light and Carrington, Adv. Chem. Phys. 114, 263 (2000)
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.sparse.linalg as sla
import scipy.special.orthogonal as ortho
import dvr_1d

class DVR(object):
    def __cartesian_product(self, arrays):
        """A fast cartesion product function that I blatantly stole from 
        user senderle on stackoverflow.com"""
        broadcastable = np.ix_(*arrays)
        broadcasted = np.broadcast_arrays(*broadcastable)
        rows, cols = reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
        out = np.empty(rows * cols, dtype=broadcasted[0].dtype)
        start, end = 0, rows
        for a in broadcasted:
            out[start:end] = a.reshape(-1)
            start, end = end, end + rows
        return out.reshape(cols, rows).T

    def __init__(self, dvr1d):
        self.dvr1d = dvr1d
        self.x = dvr1d.x
        self.y = dvr1d.x
        self.xy = np.fliplr(self.__cartesian_product([self.x, self.y]))

    def v(self, V):
        """Return the potential matrix with the given potential.
        Usage:
            v_matrix = self.v(V)

        @param[in] V potential function
        @returns v_matrix potential matrix
        """
        return np.diag(V(self.xy))

    def t(self):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t()

        @returns T kinetic energy matrix
        """
        t1d = self.dvr1d.t()
        eye = np.identity(self.dvr1d.npts)
        return np.kron(eye, t1d) + np.kron(t1d, eye)

    def h(self, V):
        """Return the hamiltonian matrix with the given potential.
        Usage:
            H = self.h(V)

        @param[in] V potential function
        @returns H potential matrix
        """
        return self.t() + self.v(V)

    def plot(self, V, E, U, **kwargs):
        doshow = kwargs.get('doshow', False)
        nplot = kwargs.get('nplot', 5)
        xmin = kwargs.get('xmin', self.xy[:,0].min())
        xmax = kwargs.get('xmax', self.xy[:,0].max())
        ymin = kwargs.get('ymin', self.xy[:,1].min())
        ymax = kwargs.get('ymax', self.xy[:,1].max())
        zmin = kwargs.get('zmin', np.ceil(V(self.xy).min() - 1.))
        zmax = kwargs.get('zmax', 
                          np.floor(max(U.max()+E.max()+1., 
                                   V(self.xy).max()+1.)))

        npts = self.dvr1d.npts
        xy = self.xy.reshape((npts, npts, 2))
        vp = V(self.xy).reshape((npts, npts))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot_wireframe(xy[:,:,0], xy[:,:,1], vp)
        for i in range(nplot):
            if i == 0:
                ax.plot_surface(xy[:,:,0], xy[:,:,1], 
                                abs(U[:, i].reshape((npts, npts))) + E[i], 
                                alpha=0.5)
            #else:
                #ax.plot_surface(xy[:,:,0], xy[:,:,1], 
                #                U[:, i].reshape((npts, npts)) + E[i], 
                #                alpha=0.5)
        #plt.axis(ymax=ymax, ymin=ymin)
        #plt.axis(xmax=xmax, xmin=xmin)
        if doshow: plt.show()
        return

    def test_potential(self, V, num_eigs = 5, **kwargs):
        h = self.h(V)
        # Get the eigenpairs
        # There are multiple options here. 
        # If the user is asking for all of the eigenvalues, 
        # then we need to use np.linalg.eigh()
        if num_eigs == h.shape[0]:
            E, U = np.linalg.eigh(h)
        # But if we don't need all eigenvalues, only the smallest ones,
        # then when the size of the H matrix becomes large enough, it is
        # better to use sla.eigsh() with a shift-invert method. Here we
        # have to have a good guess for the smallest eigenvalue so we
        # ask for eigenvalues closest to the minimum of the potential.
        else:
            E, U = sla.eigsh(h, k=num_eigs, which='LM', 
                             sigma=V(self.xy).min())

        precision = kwargs.get('precision', 8)

        # Print and plot stuff
        print 'The first {n:d} energies are:'.format(n=num_eigs)
        print np.array_str(E[:num_eigs], precision=precision)

        doshow = kwargs.get('doshow', False)
        if doshow:
            xmin = kwargs.get('xmin', self.xy[:,0].min())
            xmax = kwargs.get('xmax', self.xy[:,0].max())
            ymin = kwargs.get('ymin', self.xy[:,1].min())
            ymax = kwargs.get('ymax', self.xy[:,1].max())
            zmin = kwargs.get('zmin', np.ceil(V(self.xy).min() - 1.))
            zmax = kwargs.get('zmax', 
                              np.floor(max(U.max()+E.max()+1., 
                                       V(self.xy).max()+1.)))

            self.plot(V, E, U, nplot=num_eigs, 
                      xmin=xmin, xmax=xmax,
                      ymin=ymin, ymax=ymax, 
                      zmin=zmin, zmax=zmax,
                      doshow=doshow)
        return E, U

    def sho_test(self, num_eigs=5, precision=8, doshow=False):
        print 'Testing 2-D DVR with an SHO potential'
        vF = VFactory()
        V = vF.sho()
        E, U = self.test_potential(V, doshow=doshow, num_eigs=num_eigs, 
                                   precision=precision,
                                   xmin=-3.5, xmax=3.5, 
                                   ymin=0., ymax=6.)
        print
        return E, U

# Factory functions to build different potentials:
# A factory is a function that makes a function. 
class VFactory(object):
    """Factory functions to build different potentials
    A factory is a function that returns other functions.
    """
    # def square_well(self, depth = 1., width = 1., 
    #                 origin = 0., o_val = 0.):
    #     """Usage:
    #             V = square_well_factory(**kwargs)

    #     Returns a function of a single variable V(x), 
    #     representing the square-well potential:

    #          (-A/2, V0)            (A/2, V0)                                   
    #     ------------       +       ----------------
    #                |               |
    #                |               |
    #                |               |
    #                |               |
    #      (-A/2, 0) |-------+-------| (A/2, 0)
    #                      (0, 0)

    #     Keyword arguments:
    #     @param[in] depth    Depth of the potential well (default=1)
    #     @param[in] width    Width of the potential well (default=1)
    #     @param[in] origin   Location of the well's center (default=0)
    #     @param[in] o_val    Value of the potential at origin (default=0)
    #     @returns   V        The square well potential function V(x)
    #     """
    #     def V(x):
    #         interior_idx = np.abs(x - origin) < width / 2.
    #         V = np.ones_like(x) * (depth + o_val)
    #         V[interior_idx] = o_val 
    #         return V
    #     return V

    # def double_well(self, x1 = -2., x2 = -1., x3 = 1., 
    #                 x4 = 2., V1 = 1., V2 = 0., 
    #                 V3 = 1., V4 = 0., V5 = 1.):
    #     """Usage:
    #             V = double_square_well_factory(**kwargs)

    #     Returns a one-dimensional potential function that represents
    #     a double-square-well potential. The potential looks like

    #        (x1, V1)      (x2, V3)   (x3, V3)      (x4, V5)
    #     ----------            ---------            ---------- 
    #              |            |       |            |
    #              |            |       |            |
    #              |            |       |            |
    #              |            |       |            |
    #              |____________|       |____________|
    #        (x1, V2)      (x2, V2)   (x3, V4)      (x4, V4)

    #     Keywork arguments
    #     @param[in] x1    x-coordinate x1 above (default=-2)
    #     @param[in] x2    x-coordinate x2 above (default=-1)
    #     @param[in] x3    x-coordinate x3 above (default=1)
    #     @param[in] x4    x-coordinate x4 above (default=2)
    #     @param[in] V1    constant V1 above (default=1)
    #     @param[in] V2    constant V2 above (default=0)
    #     @param[in] V3    constant V3 above (default=1)
    #     @param[in] V4    constant V4 above (default=0)
    #     @param[in] V5    constant V5 above (default=1)
    #     @returns   V     double square-well potential V(x)
    #     """
    #     assert (x1 < x2 < x3 < x4), \
    #         "x-coordinates do not satisfy x1 < x2 < x3 < x4"
    #     def V(x):
    #         l_well_idx = np.logical_and(x < x2, x > x1)
    #         r_well_idx = np.logical_and(x < x4, x > x3)
    #         middle_idx = np.logical_and(x >= x2, x <= x3)
    #         far_rt_idx = np.greater_equal(x, x4)
    #         V = np.ones_like(x) * V1
    #         V[l_well_idx] = V2
    #         V[middle_idx] = V3
    #         V[r_well_idx] = V4
    #         V[far_rt_idx] = V5
    #         return V
    #     return V

    def sho(self, k = 1., x0 = 0., y0 = 0.):
        """Usage:
                V = harmosc_factory(**kwargs)
     
        Return a two-dimensional harmonic oscillator potential V(x, y)
        with wavenumber k. 
        i.e. V(x, y) = 1/2 * k * ((x - x0)^2 + (y - y0)^2)

        Keyword arguments
        @param[in] k    wavenumber of the SHO potential (default=1)
        @param[in] x0   x-displacement from origin (default=0)
        @param[in] y0   y-displacement from origin (default=0)
        @returns   V    2-D SHO potential V(x)
        """
        def V(xy): return 0.5 * k * (np.square(xy[:,0] - x0) 
                                   + np.square(xy[:,1] - y0))
        return V

    # def power(self, a = 1., p=1., x0 = 0.):
    #     """Usage:
    #             V = self.power(**kwargs)

    #     Return a potential V(x) = a * (x - x0)^p

    #     Keyword arguments
    #     @param[in] a    coefficient (default=1)
    #     @param[in] p    power to raise x (default=1)
    #     @param[in] x0   displacement from origin (default=0)
    #     @returns   V    1-D cubic potential V(x)
    #     """
    #     def V(x): return a * np.power(x - x0, p)
    #     return V

    # def morse(self, D = 1., a = 1., x0 = 0.):
    #     """Usage:
    #             V = morse_factory(**kwargs)

    #     Return a one-dimensional Morse potential V(x)
    #     i.e. V(x) = D * (1 - exp(-a * (x - x0)))^2 - D

    #     Keyword arguments
    #     @param[in] D    dissociation depth
    #     @param[in] a    inverse "width" of the potential
    #     @param[in] x0   equilibrium bond distance
    #     @returns   V    Morse potential V(x)
    #     """
    #     def V(x): 
    #         return D * np.power(1. - np.exp(-a * (x - x0)), 2.) - D
    #     return V

    # def sombrero(self, a = -10., b = 1.):
    #     """Usage:
    #             V = sombrero_factory(**kwargs)
     
    #     Return a one-dimensional version of the sombrero potential
    #     i.e. V(x) = a * x^2 + b * x^4
    #     This function asserts a < 0 and b > 0

    #     Keyword arguments
    #     @param[in] a    coefficient of the x^2 term (default=-10)
    #     @param[in] b    coefficient of the x^4 term (default=1)
    #     @returns   V    1-D Mexican hat potential V(x)
    #     """
    #     assert (a < 0), "Coefficient a must be negative"
    #     assert (b > 0), "Coefficient b must be positive"
    #     def V(x):
    #         return a * np.square(x) + b * np.power(x, 4)
    #     return V

    # def woods_saxon(self, V0 = 50., z = 0.5, r0 = 1.2, A = 16):
    #     """Usage:
    #             V = woods_saxon_factory(**kwargs)
     
    #     Return a Woods-Saxon potential
    #     i.e. V(r) = - V0 / (1. + exp((r - R) / z))
    #     where R = r0 * A^(1/3)

    #     Keyword arguments
    #     @param[in] V0   potential depth (default=50.)
    #     @param[in] z    surface thickness (default=0.5)
    #     @param[in] r0   rms nuclear radius (default=1.2)
    #     @param[in] A    mass number (default=16)
    #     @returns   V    Woods-Saxon potential V(r)
    #     """
    #     def V(r): 
    #         x0 = r0 * np.power(A, 1. / 3.)
    #         return -V0 / (1. + np.exp((r - x0)/ z))
    #     return V

