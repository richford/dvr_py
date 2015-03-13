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
import scipy.sparse as sp
import scipy.special.orthogonal as ortho
import dvr_1d

# These are the "Tableau 20" colors as RGB.  
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# These are the "Tableau 10 Medium" colors as RGB.  
tableau10m = [(114, 158, 206), (255, 158, 74), (103, 191, 92), (237, 102, 93), 
              (173, 139, 201), (168, 120, 110), (237, 151, 202), 
              (162, 162, 162), (205, 204, 93), (109, 204, 218)]

# These are the Tableau "Color Blind 10" colors as RGB
colorblind10 = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89), 
                (95, 158, 209), (200, 82, 0), (137, 137, 137), (162, 200, 236), 
                (255, 188, 121), (207, 207, 207)]


# Scale the RGB values to [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)
for i in range(len(tableau10m)):
    r, g, b = tableau10m[i]  
    tableau10m[i] = (r / 255., g / 255., b / 255.)
    r, g, b = colorblind10[i]  
    colorblind10[i] = (r / 255., g / 255., b / 255.)

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

    def __init__(self, dvr1d, spf='csr'):
        self.dvr1d = dvr1d
        self.x = dvr1d.x
        self.y = dvr1d.x
        self.z = dvr1d.x
        self.xyz = np.fliplr(self.__cartesian_product([self.x, self.y, self.z]))
        self.spf = spf

    def v(self, V):
        """Return the potential matrix with the given potential.
        Usage:
            v_matrix = self.v(V)

        @param[in] V potential function
        @returns v_matrix potential matrix
        """
        return sp.diags(diagonals=V(self.xyz), offsets=0, format=self.spf)

    def t(self):
        """Return the kinetic energy matrix.
        Usage:
            T = self.t()

        @returns T kinetic energy matrix
        """
        t1d = self.dvr1d.t()
        eye = sp.identity(self.dvr1d.npts, format=self.spf)
        t2d = sp.kron(eye, t1d, format=self.spf) + sp.kron(t1d, eye, format=self.spf)
        return sp.kron(t2d, eye, format=self.spf) + sp.kron(eye, sp.kron(eye, t1d), format=self.spf)

    def h(self, V):
        """Return the hamiltonian matrix with the given potential.
        Usage:
            H = self.h(V)

        @param[in] V potential function
        @returns H potential matrix
        """
        return self.t() + self.v(V)

    def plot(self, V, E, U, **kwargs):
        assert False, "plotting not yet implemented for 3D-DVR"
        pass

    def test_potential(self, V, num_eigs = 5, **kwargs):
        h = self.h(V)
        # Get the eigenpairs
        # There are multiple options here. 
        # If the user is asking for all of the eigenvalues, 
        # or if for some reason they want to use np.eigh
        # then we need to use np.linalg.eigh()
        do_full_eig = kwargs.get('do_full_eig', False)
        if do_full_eig or (num_eigs == h.shape[0]): 
            print h.ndim
            print len(h.shape)
            E, U = np.linalg.eigh(h)
        # But if we don't need all eigenvalues, only the smallest ones,
        # then when the size of the H matrix becomes large enough, it is
        # better to use sla.eigsh() with a shift-invert method. Here we
        # have to have a good guess for the smallest eigenvalue so we
        # ask for eigenvalues closest to the minimum of the potential.
        else:
            E, U = sla.eigsh(h, k=num_eigs, which='LM', 
                             sigma=V(self.xyz).min())

        precision = kwargs.get('precision', 8)

        # Print and plot stuff
        print 'The first {n:d} energies are:'.format(n=num_eigs)
        print np.array_str(E[:num_eigs], precision=precision)

        doshow = kwargs.get('doshow', False)
        assert doshow==False, \
                'Plotting is not yet implemented. Please use doshow=False'
        if doshow:
            uscale = kwargs.get('uscale', 1.)
            xmin = kwargs.get('xmin', self.xyz[:,0].min())
            xmax = kwargs.get('xmax', self.xyz[:,0].max())
            ymin = kwargs.get('ymin', self.xyz[:,1].min())
            ymax = kwargs.get('ymax', self.xyz[:,1].max())
            zmin = kwargs.get('zmin', np.ceil(V(self.xyz).min() - 1.))
            zmax = kwargs.get('zmax', 
                              np.floor(max(U.max()+E.max()+1., 
                                       V(self.xyz).max()+1.)))

            self.plot(V, E, U, nplot=num_eigs, 
                      xmin=xmin, xmax=xmax,
                      ymin=ymin, ymax=ymax, 
                      zmin=zmin, zmax=zmax,
                      uscale=uscale, doshow=doshow)
        return E, U

    def sho_test(self, k = 1., num_eigs=5, precision=8, 
                 uscale=1., doshow=False, do_full_eig=False):
        print 'Testing 3-D DVR with an SHO potential'
        vF = VFactory()
        V = vF.sho(k=k)
        E, U = self.test_potential(V, doshow=doshow, num_eigs=num_eigs, 
                                   precision=precision, uscale=uscale,
                                   do_full_eig=do_full_eig,
                                   xmin=-3.5, xmax=3.5, 
                                   ymin=-3.5, ymax=3.5,
                                   zmin=-0.05, zmax=4.)
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

    def sho(self, k = 1., x0 = 0., y0 = 0., z0 = 0.):
        """Usage:
                V = harmosc_factory(**kwargs)
     
        Return a two-dimensional harmonic oscillator potential V(x, y)
        with wavenumber k. 
        i.e. V(x, y) = 1/2 * k * ((x - x0)^2 + (y - y0)^2)

        Keyword arguments
        @param[in] k    wavenumber of the SHO potential (default=1)
        @param[in] x0   x-displacement from origin (default=0)
        @param[in] y0   y-displacement from origin (default=0)
        @param[in] z0   z-displacement from origin (default=0)
        @returns   V    2-D SHO potential V(x)
        """
        def V(xyz): return 0.5 * k * (np.square(xyz[:,0] - x0) 
                                    + np.square(xyz[:,1] - y0)
                                    + np.square(xyz[:,2] - z0))
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

