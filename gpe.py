import dvr_1d
import numpy as np
from scipy.optimize import newton_krylov
import scipy as sp

class GPESolver(object):
    '''
    Problem with Gross-Pitaevskii Equation 
    '''
    def __init__(self, v_trap, g=-1.0, N=1.0, npts=100, R=10.0, dim=3, lam=0.0):
        self.npts = npts
        self.R = float(R)
        self.dim = int(dim)
        self.lam = lam
        self.basis = dvr_1d.BesselDVR(npts=npts, R=R, dim=dim, lam=lam);
        self.g = float(g)  # Intensity of the interaction coupling constant!
        self.N = float(N)  # particle number!
        self.v_trap = v_trap

    @property
    def v(self):
        return self.v_trap(self.basis.x)

    @property
    def metric(self):
        if self.dim == 2:
            return 2.*np.pi*self.basis.x
        if self.dim == 3:
            return 4.*np.pi*self.basis.x**2.
    
    def bracket(self, a, b):
        # a, b are radial wave function like vectors!
        metric = self.metric
        return np.trapz(a*b*metric/self.basis.x**(self.dim-1.), self.basis.x)

    def normalize(self,y):
        # normalize the wavefunctions!
        # y is radial wf!
        return y * np.sqrt(self.N) / np.sqrt(self.bracket(y,y))

    def get_mu(self,y):
        # get chemical potential of a given wave functions!
        Hy = self.get_Hy(y)
        mu = self.bracket(Hy,y)/self.bracket(y,y)
        return mu

    def get_Hy(self,y):
        # y is radial wave function
        # y should be normalized before using!
        T = self.basis.t();
        Hy= np.dot(T,y) + self.v*y+ self.g*y**2./self.basis.x**(self.dim-1)*y
        return Hy

    def get_energy(self, y):
        # calculate the total energy 
        # y is radial wave functions!
        psi = y/self.basis.x**((self.dim-1.)/2.); # get the actual wf!
        T = self.basis.t();
        E_kin = self.bracket(y, np.dot(T,y))  # Caculate Kinetic energy(total)
        e_ext = self.v*abs(psi)**2.  # external potential energy
        e_int = 1./2.*self.g*abs(psi)**4.  # interaction energy
        #----calculate the total energy
        E_total = np.trapz((e_ext+e_int)*self.metric, self.basis.x) + E_kin; 
        return E_total 

    def get_psi(self,y):
        # return the actual wave function!
        return y/self.basis.x**((self.dim-1.)/2.)

    def solve(self, xin=None):
        '''
        input:
        N: particle number
        g: intensity of the interations
        #########
        output:
        y: the radial wave function
        mu: the chemical potential
        '''
        r = self.basis.x

        def get_dy(_y):
            y  = self.normalize(_y)
            dy = self.get_Hy(y)
            mu = self.get_mu(y)
            dy -= mu*y
            return dy

        if xin is None:  # initial guess for the radial wave function
            xin = np.sqrt(self.N)*np.exp(-r**2)*r**((self.dim-1)/2)
        y = newton_krylov(get_dy, xin)
        mu = self.get_mu(y);
        y = self.normalize(y)
        return y, mu

if __name__=="__main__":
    import matplotlib.pyplot as plt

    def v_trap(r):
        return 0.5 * r**2.

    p = GPESolver(v_trap=v_trap, g=1., N=1., npts=100, R=10.0, dim=3, lam=0.)
    y, mu = p.solve()
    psi = p.get_psi(y)
    plt.plot(p.basis.x, p.get_psi(y))
    print('mu = ', mu)

