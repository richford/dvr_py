"""
Use a simple Discrete Variable Representation method to solve
one-dimensional potentials.

A good general introduction to DVR methods is
Light and Carrington, Adv. Chem. Phys. 114, 263 (2000)
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse.linalg as sla
import scipy.special.orthogonal as ortho
import warnings

class DVR(object):
    def dvr2fbr(DVR, T):
        """Transform a matrix from the discrete variable representation
        to the finite basis representation"""
        return np.dot(T, np.dot(DVR, np.transpose(T)))

    def fbr2dvr(FBR, T):
        """Transform a matrix from the finite basis representation to the
        discrete variable representation."""
        return np.dot(np.transpose(T), np.dot(FBR, T))

    def plot(self, V, E, U, **kwargs):
        doshow = kwargs.get('doshow', False)
        nplot = kwargs.get('nplot', 5)
        xmin = kwargs.get('xmin', self.x.min())
        xmax = kwargs.get('xmax', self.x.max())
        ymin = kwargs.get('ymin', np.ceil(V(self.x).min() - 1.))
        ymax = kwargs.get('ymax', 
                          np.floor(max(U.max()+E.max()+1., V(self.x).max()+1.)))
        plt.plot(self.x, V(self.x))
        for i in range(nplot):
            if i == 0:
                plt.plot(self.x, abs(U[:, i])+E[i])
            else:
                plt.plot(self.x, U[:, i]+E[i])
        plt.axis(ymax=ymax, ymin=ymin)
        plt.axis(xmax=xmax, xmin=xmin)
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
                             sigma=V(self.x).min())

        xmin = kwargs.get('xmin', self.x.min())
        xmax = kwargs.get('xmax', self.x.max())
        ymin = kwargs.get('ymin', np.ceil(V(self.x).min() - 1.))
        ymax = kwargs.get('ymax', 
                          np.floor(max(U.max()+E.max()+1., V(self.x).max()+1.)))
        precision = kwargs.get('precision', 8)

        # Print and plot stuff
        print 'The first {n:d} energies are:'.format(n=num_eigs)
        print np.array_str(E[:num_eigs], precision=precision)
        self.plot(V, E, U, nplot=num_eigs, 
                  xmin=xmin, xmax=xmax,
                  ymin=ymin, ymax=ymax, 
                  doshow=True)

        return

    def square_well_test(self, precision=8):
        print 'Testing 1-D DVR with a square-well potential'
        vF = VFactory()
        V = vF.square_well(depth=9./2., width=10.)
        self.test_potential(V, num_eigs=5, precision=precision, 
                            xmin=-10., xmax=10., 
                            ymin=-0.25, ymax=2.)
        return

    def double_well_test(self, precision=8):
        print 'Testing 1-D DVR with a double-well potential'
        vF = VFactory()
        V = vF.double_well()
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=-3.5, xmax=3.5, 
                            ymin=-0.5, ymax=4.)
        return

    def sho_test(self, precision=8):
        print 'Testing 1-D DVR with an SHO potential'
        vF = VFactory()
        V = vF.sho()
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=-3.5, xmax=3.5, 
                            ymin=0., ymax=6.)
        return

    def morse_test(self, precision=8):
        print 'Testing 1-D DVR with a Morse potential'
        vF = VFactory()
        V = vF.morse(D=3., a=0.5)
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=-3., xmax=32., 
                            ymin=-3., ymax=1.)
        return

    def sombrero_test(self, precision=8):
        print 'Testing 1-D DVR with a sombrero potential'
        vF = VFactory()
        V = vF.sombrero(a=-5.)
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=-5., xmax=5., ymax=5.)
        return

    def woods_saxon_test(self, precision=8):
        print 'Testing 1-D DVR with a Woods-Saxon potential'
        vF = VFactory()
        V = vF.woods_saxon(A=4)
        self.test_potential(V, num_eigs=5, precision=precision,
                            xmin=0., xmax=5., 
                            ymin=-50., ymax=0.)
        return

    def test_all(self, precision=8):
        self.square_well_test(precision=precision)
        self.double_well_test(precision=precision)
        self.sho_test(precision=precision)
        self.morse_test(precision=precision)
        self.sombrero_test(precision=precision)
        self.woods_saxon_test(precision=precision)

class SincDVR(DVR):
    r"""Sinc function basis for non-periodic functions over an interval
    `x0 +- L/2` with `N` points.
    Usage:
        d = sincDVR1D(N, L, [x0])

    @param[in] N number of points
    @param[in] L size of interval
    @param[in] x0 origin offset (default=0)
    @attribute a step size
    @attribute n vector of x-domain indices
    @attribute x discretized x-domain
    @attribute k_max cutoff frequency
    @method h return hamiltonian matrix
    @method f return DVR basis vectors
    """
    def __init__(self, npts, L, x0=0.):
        L = float(L)
        self.npts = npts
        self.L = L
        self.x0 = x0
        self.a = L / npts
        self.n = np.arange(npts)
        self.x = self.x0 + self.n * self.a - self.L / 2. + self.a / 2.
        self.w = np.ones(npts, dtype=np.float64) * self.a
        self.k_max = np.pi/self.a

    def h(self, V):
        """Return the Hamiltonian with the given potential.
        Usage:
            H = self.h(V)

        @param[in] V potential
        """
        _m = self.n[:, None]
        _n = self.n[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K = 2. * (-1.)**(_m-_n) / (_m-_n)**2. / self.a**2.
        K[self.n, self.n] = np.pi**2. / 3. / self.a**2.
        K *= 0.5   # p^2/2/m
        V = np.diag(V(self.x))
        return K + V

    def f(self, x=None):
        """Return the DVR basis vectors"""
        if x is None:
            x_m = self.x[:, None]
        else:
            x_m = np.asarray(x)[:, None]
        x_n = self.x[None, :]
        return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)

class SincDVRPeriodic(SincDVR):
    r"""Sinc function basis for periodic functions over an interval
    `x0 +- L/2` with `N` points."""
    def __init__(self, *v, **kw):
        # Small shift here for consistent abscissa
        SincDVR.__init__(self, *v, **kw)
        self.x -= self.a/2.
        
    def h(self, V):
        """Return the Hamiltonian with the given potential.
        Usage:
            H = self.h(V)

        @param[in] V potential
        """
        _m = self.n[:, None]
        _n = self.n[None, :]
        _arg = np.pi*(_m-_n)/self.npts
        if (0 == self.npts % 2):
            K = 2.*(-1.)**(_m-_n)/np.sin(_arg)**2.
            K[self.n, self.n] = (self.npts**2. + 2.)/3.
        else:
            K = 2.*(-1.)**(_m-_n)*np.cos(_arg)/np.sin(_arg)**2.
            K[self.n, self.n] = (self.npts**2. - 1.)/3.
        K *= 0.5*(np.pi/self.L)**2.   # p^2/2/m
        V = np.diag(V(self.x))
        return K + V

    def f(self, x=None):
        """Return the DVR basis vectors"""
        if x is None:
            x_m = self.x[:, None]
        else:
            x_m = np.asarray(x)[:, None]
        x_n = self.x[None, :]
        f = np.sinc((x_m-x_n)/self.a)/np.sinc((x_m-x_n)/self.L)/np.sqrt(self.a)
        if (0 == self.npts % 2):
            f *= np.exp(-1j*np.pi*(x_m-x_n)/self.L)
        return f

# class FourierDVR(DVR):
#     def __init__(self, npts, x0=0.):
#         L = float(L)
#         self.npts = npts
#         self.L = L
#         self.x0 = x0
#         self.a = L / npts
#         self.n = np.arange(npts, dtype=np.float64)
#         self.x = np.arange(-0.5 * (npts - 1), 0.5 * npts, dtype=np.float64)
#         self.w = np.ones(npts, dtype=np.float64)
#         self.k_max = np.pi/self.a

#     def h(self, V):
#         """Return the Hamiltonian with the given potential.
#         Usage:
#             H = self.h(V)

#         @param[in] V potential
#         """
#         _m = self.n[:, None]
#         _n = self.n[None, :]
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             K = 2.*(-1)**(_m-_n)/(_m-_n)**2/self.a**2
#         K[self.n, self.n] = np.pi**2/3/self.a**2
#         K *= 0.5   # p^2/2/m
#         V = np.diag(V(self.x))
#         return K + V

#     def f(self, x=None):
#         """Return the DVR basis vectors"""
#         if x is None:
#             x_m = self.x[:, None]
#         else:
#             x_m = np.asarray(x)[:, None]
#         x_n = self.x[None, :]
#         return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)

class SineDVR(DVR):
    def __init__(self, npts, x0=0.):
        self.npts = npts
        self.x0 = float(x0)
        self.n = np.arange(npts)
        self.x = np.arange(-0.5*(npts - 1), 0.5*npts, dtype=np.float64)
        self.a = float(npts) / (float(npts) + 1.)
        self.x *= self.a
        self.x += self.x0
        self.L = self.x.max() - self.x.min()
        self.k_max = None

    def h(self, V):
        """Return the Hamiltonian with the given potential.
        Usage:
            H = self.h(V)

        @param[in] V potential
        """
        _i = self.n[:, None]
        _j = self.n[None, :]
        _xi = self.x[:, None] * np.pi / (2. * self.npts)
        _xj = self.x[None, :] * np.pi / (2. * self.npts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K = ((-1.)**(_i-_j) / self.npts**2.
                * (1./np.square(np.sin(_xi-_xj)) 
                   - 1./np.square(np.cos(_xi + _xj))))
        K[self.n, self.n] = 0.
        K += np.diag(2./3. * (self.npts + 1.)**2. + 1./3.
                     - np.power(np.cos(np.pi * self.x / self.npts), -2.)) / (self.npts**2.)
        K *= np.pi**2. / 2.
        K *= 0.5   # p^2/2/m
        V = np.diag(V(self.x))
        return K + V

#     def f(self, x=None):
#         """Return the DVR basis vectors"""
#         if x is None:
#             x_m = self.x[:, None]
#         else:
#             x_m = np.asarray(x)[:, None]
#         x_n = self.x[None, :]
#         return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)

class HermiteDVR(DVR):
    def __init__(self, npts, x0=0.):
        assert (npts < 269), \
            "Must make npts < 269 for python to find quadrature points."
        self.npts = npts
        self.x0 = float(x0)
        self.n = np.arange(npts)
        c = np.zeros(npts+1)
        c[-1] = 1.
        self.x = np.polynomial.hermite.hermroots(c)
        self.x += self.x0
        self.w = np.exp(-np.square(self.x))
        self.L = self.x.max() - self.x.min()
        self.a = None
        self.k_max = None

    def h(self, V):
        """Return the Hamiltonian with the given potential.
        Usage:
            H = self.h(V)

        @param[in] V potential
        """
        _i = self.n[:, None]
        _j = self.n[None, :]
        _xi = self.x[:, None]
        _xj = self.x[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K = 2.*(-1.)**(_i-_j)/(_xi-_xj)**2.
        K[self.n, self.n] = 0.
        K += np.diag((2. * self.npts + 1. 
                      - np.square(self.x)) / 3.)
        K *= 0.5   # p^2/2/m
        V = np.diag(V(self.x))
        return K + V

#     def f(self, x=None):
#         """Return the DVR basis vectors"""
#         if x is None:
#             x_m = self.x[:, None]
#         else:
#             x_m = np.asarray(x)[:, None]
#         x_n = self.x[None, :]
#         return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)

# class GaussianDVR(DVR):
#     def __init__(self, npts, x0=0.):
#         assert (npts < 269), \
#             "Must make npts < 269 for python to find quadrature points."
#         self.npts = npts
#         self.x0 = float(x0)
#         self.n = np.arange(npts)
#         c = np.zeros(npts+1)
#         c[-1] = 1.
#         self.x = np.polynomial.hermite.hermroots(c)
#         self.x += self.x0
#         self.w = np.exp(-np.square(self.x))
#         self.L = self.x.max() - self.x.min()
#         self.a = None
#         self.k_max = None

#     def h(self, V):
#         """Return the Hamiltonian with the given potential.
#         Usage:
#             H = self.h(V)

#         @param[in] V potential
#         """
#         _i = self.n[:, None]
#         _j = self.n[None, :]
#         _m = self.x[:, None]
#         _n = self.x[None, :]
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             K = 2.*(-1)**(_i-_j)/(_m-_n)**2.
#         K[self.n, self.n] = 0.
#         K += np.diag((2. * self.npts + 1. 
#                       - np.square(self.x)) / 3.)
#         K *= 0.5   # p^2/2/m
#         V = np.diag(V(self.x))
#         return K + V

#     def f(self, x=None):
#         """Return the DVR basis vectors"""
#         if x is None:
#             x_m = self.x[:, None]
#         else:
#             x_m = np.asarray(x)[:, None]
#         x_n = self.x[None, :]
#         return np.sinc((x_m-x_n)/self.a)/np.sqrt(self.a)

# class ChebDVR(DVR):
#     def __init__(self, npts, xmin, xmax):
#         delta = float(xmax) - float(xmin)
#         self.npts = npts
#         self.L = delta
#         self.x0 = (float(xmin) + float(xmax)) / 2.
#         self.a = delta / npts
#         self.n = np.arange(1, npts + 1, dtype=np.float64)
#         self.x = float(xmin) + self.n * delta / (npts + 1.)
#         self.w = np.ones(npts, dtype=np.float64) * delta / (npts + 1.)
#         self.k_max = np.pi/self.a

#     def h(self, V):
#         """Return the Hamiltonian matrix with the given potential
#         Usage:
#                 H = self.h(V)

#         Returns the quadrature points, the transformation matrix, the
#         kinetic energy operator, and the quadrature weights corresponding
#         to a Chebyshev-based discrete variable representation.

#         @param[in] V potential
#         @returns H hamiltonian
#         """
#         # Work in more "natural" units
#         h = m = 1.
#         K = np.square(self.n * np.pi / delta) * h * h / (2. * m)
#         K = np.diag(K)
#         V = np.diag(V(self.x))
#         return K + V

#     def f(self, x=None):
#         """Return the DVR basis vectors"""
#         if x is None:
#             x_m = self.x[:, None]
#         else:
#             x_m = np.asarray(x)[:, None]
#         x_n = self.x[None, :]
#         return 1
#     """
#     Transformation matrix stuff
#     iv, jv = np.meshgrid(ivec, ivec)
#     T = np.sin(iv * jv * np.pi / (npts + 1.))
#     T = T * np.sqrt(2 / (npts + 1.))
#     """


# Factory functions to build different potentials:
# A factory is a function that makes a function. 
class VFactory(object):
    """Factory functions to build different potentials
    A factory is a function that returns other functions.
    """
    def __init__(self):
        pass

    def square_well(self, depth = 1., width = 1., 
                    origin = 0., o_val = 0.):
        """Usage:
                V = square_well_factory(**kwargs)

        Returns a function of a single variable V(x), 
        representing the square-well potential:

             (-A/2, V0)            (A/2, V0)                                   
        ------------       +       ----------------
                   |               |
                   |               |
                   |               |
                   |               |
         (-A/2, 0) |-------+-------| (A/2, 0)
                         (0, 0)

        Keyword arguments:
        @param[in] depth    Depth of the potential well (default=1)
        @param[in] width    Width of the potential well (default=1)
        @param[in] origin   Location of the well's center (default=0)
        @param[in] o_val    Value of the potential at origin (default=0)
        @returns   V        The square well potential function V(x)
        """
        def V(x):
            interior_idx = np.abs(x - origin) < width / 2.
            V = np.ones_like(x) * (depth + o_val)
            V[interior_idx] = o_val 
            return V
        return V

    def double_well(self, x1 = -2., x2 = -1., x3 = 1., 
                    x4 = 2., V1 = 1., V2 = 0., 
                    V3 = 1., V4 = 0., V5 = 1.):
        """Usage:
                V = double_square_well_factory(**kwargs)

        Returns a one-dimensional potential function that represents
        a double-square-well potential. The potential looks like

           (x1, V1)      (x2, V3)   (x3, V3)      (x4, V5)
        ----------            ---------            ---------- 
                 |            |       |            |
                 |            |       |            |
                 |            |       |            |
                 |            |       |            |
                 |____________|       |____________|
           (x1, V2)      (x2, V2)   (x3, V4)      (x4, V4)

        Keywork arguments
        @param[in] x1    x-coordinate x1 above (default=-2)
        @param[in] x2    x-coordinate x2 above (default=-1)
        @param[in] x3    x-coordinate x3 above (default=1)
        @param[in] x4    x-coordinate x4 above (default=2)
        @param[in] V1    constant V1 above (default=1)
        @param[in] V2    constant V2 above (default=0)
        @param[in] V3    constant V3 above (default=1)
        @param[in] V4    constant V4 above (default=0)
        @param[in] V5    constant V5 above (default=1)
        @returns   V     double square-well potential V(x)
        """
        assert (x1 < x2 < x3 < x4), \
            "x-coordinates do not satisfy x1 < x2 < x3 < x4"
        def V(x):
            l_well_idx = np.logical_and(x < x2, x > x1)
            r_well_idx = np.logical_and(x < x4, x > x3)
            middle_idx = np.logical_and(x >= x2, x <= x3)
            far_rt_idx = np.greater_equal(x, x4)
            V = np.ones_like(x) * V1
            V[l_well_idx] = V2
            V[middle_idx] = V3
            V[r_well_idx] = V4
            V[far_rt_idx] = V5
            return V
        return V

    def sho(self, k = 1., x0 = 0.):
        """Usage:
                V = harmosc_factory(**kwargs)
     
        Return a one-dimensional harmonic oscillator potential V(x)
        with wavenumber k. i.e. V(x) = 1/2 * k * (x - x0)^2

        Keyword arguments
        @param[in] k    wavenumber of the SHO potential (default=1)
        @param[in] x0   displacement from origin (default=0)
        @returns   V    1-D SHO potential V(x)
        """
        def V(x): return 0.5 * k * np.square(x - x0)
        return V

    def power(self, a = 1., p=1., x0 = 0.):
        """Usage:
                V = self.power(**kwargs)

        Return a potential V(x) = a * (x - x0)^p

        Keyword arguments
        @param[in] a    coefficient (default=1)
        @param[in] p    power to raise x (default=1)
        @param[in] x0   displacement from origin (default=0)
        @returns   V    1-D cubic potential V(x)
        """
        def V(x): return a * np.power(x - x0, p)
        return V

    def morse(self, D = 1., a = 1., x0 = 0.):
        """Usage:
                V = morse_factory(**kwargs)

        Return a one-dimensional Morse potential V(x)
        i.e. V(x) = D * (1 - exp(-a * (x - x0)))^2 - D

        Keyword arguments
        @param[in] D    dissociation depth
        @param[in] a    inverse "width" of the potential
        @param[in] x0   equilibrium bond distance
        @returns   V    Morse potential V(x)
        """
        def V(x): 
            return D * np.power(1. - np.exp(-a * (x - x0)), 2.) - D
        return V

    def sombrero(self, a = -10., b = 1.):
        """Usage:
                V = sombrero_factory(**kwargs)
     
        Return a one-dimensional version of the sombrero potential
        i.e. V(x) = a * x^2 + b * x^4
        This function asserts a < 0 and b > 0

        Keyword arguments
        @param[in] a    coefficient of the x^2 term (default=-10)
        @param[in] b    coefficient of the x^4 term (default=1)
        @returns   V    1-D Mexican hat potential V(x)
        """
        assert (a < 0), "Coefficient a must be negative"
        assert (b > 0), "Coefficient b must be positive"
        def V(x):
            return a * np.square(x) + b * np.power(x, 4)
        return V

    def woods_saxon(self, V0 = 50., z = 0.5, r0 = 1.2, A = 16):
        """Usage:
                V = woods_saxon_factory(**kwargs)
     
        Return a Woods-Saxon potential
        i.e. V(r) = - V0 / (1. + exp((r - R) / z))
        where R = r0 * A^(1/3)

        Keyword arguments
        @param[in] V0   potential depth (default=50.)
        @param[in] z    surface thickness (default=0.5)
        @param[in] r0   rms nuclear radius (default=1.2)
        @param[in] A    mass number (default=16)
        @returns   V    Woods-Saxon potential V(r)
        """
        def V(r): 
            x0 = r0 * np.power(A, 1. / 3.)
            return -V0 / (1. + np.exp((r - x0)/ z))
        return V


if __name__ == '__main__':
    test_all()
