r"""Some utilities for computing properties of the Bessel functions for the DVR
basis.

Attribution: Michael Forbes, http://faculty.washington.edu/mforbes/"""
from __future__ import division

__all__ = ['sinc', 'J', 'j_root', 'J_sqrt_pole']

import numpy as np
from numpy import inf, pi, finfo
from numpy import sqrt

import scipy.special
sp = scipy

_EPS = finfo(np.double).eps
_TINY = finfo(np.double).tiny


def sinc(x, n=0):
    r"""Return the `n`'th derivative of `sinc(x) = sin(x)/x`.

    Parameters
    ----------
    x : {float, array}
       Argument
    n : int, optional
       Order of derivative.

    Examples
    --------
    >>> N = 3
    >>> np.allclose(sinc(2.0), np.sin(2.0)/2.0)
    True
    >>> print("skip ...");from mmf.math.differentiate import differentiate
    skip ...
    >>> x = np.array([-1,0,0.5,1])
    >>> np.allclose(differentiate(lambda x:sinc(x), x),
    ...             sinc(x, n=1))
    True
    """
    if 0 == n:
        return np.sinc(x/np.pi)
    elif 1 == n:
        x2 = x*x
        return np.where(abs(x) < 0.01,
                        x*(x2*(-x2/280 + 0.1) - 1.0)/3.0,
                        (np.cos(x)-np.sinc(x/np.pi))/x)
    else:
        raise NotImplementedError("Only `n=0` or `1` supported.")


def J(nu, n=0):
    r"""Return the `n`'th derivative of the bessel functions
    :math:`J_{\nu}(z)`.


    Parameters
    ----------
    nu : float
       Order.
    n : int
       Compute the `n`'th derivative.

    Examples
    --------
    >>> J0 = J(0.5); J1 = J(1.5); J2 = J(2.5);
    >>> z = 2.5; nu = 1.5
    >>> abs(J0(z) + J2(z) - 2*nu/z*J1(z)) < _EPS
    True

    .. todo:: Fix tolerances so that these are computed to machine precision.
    """
    nu2 = 2*nu
    if 0 == n:
        if 1 == nu2:
            def j(z):
                return np.sqrt(2*z/pi)*sinc(z)
        elif 3 == nu2:
            def j(z):
                return np.sqrt(2/z/pi)*(sinc(z) - np.cos(z))
        elif 5 == nu2:
            def j(z):
                return np.sqrt(2/z/pi)/z*((3.0 - z*z)*sinc(z) - 3*np.cos(z))
        elif False:                     # pragma: no cover
            def j(z):
                return 2*(nu - 1)/z*J(nu - 1)(z) - J(nu - 2)(z)
        else:
            def j(z):
                return sp.special.jn(nu, z)
    else:
        # Compute derivatives using recurrence relations.  Not
        # efficient for high orders!
        def j(z):
            return (J(nu - 1, n - 1)(z) - J(nu + 1, n - 1)(z))/2.0
    return j


def j_root_x(nu, x, rel_tol=2*_EPS):
    r"""Return the roots of the bessel function closest to `x` found
    by iterating a version of Newton's method.

    Parameters
    ----------
    nu : float
       Order of bessel function
    N : int
       Number of roots
    rel_tol : float, optional
       Desired relative tolerance for roots.
    """
    if True:
        # Algorithm from
        # Numerical Algorithms 18 (1998) 259-276
        old_err = 10
        err = 1
        n_iter = 0
        while np.any(x > nu) and err > rel_tol:
            n_iter += 1
            h = J(nu=nu)(x)/J(nu=nu-1)(x)
            #h = J_J(nu=nu, x=x)
            h = np.where(np.abs(h) > 1, np.sign(h), h)
            x_a = x
            x = x - h/(1 + h*h)
            x = np.where(x < 0, x_a/2, x)
            old_err = err
            err = np.max(np.abs(h/x))
            if err >= old_err and n_iter > 20:  # pragma: no cover
                warn("j_root: terminating iteration with error " +
                     "%g < %g less that specified rel_tol"
                     % (err, rel_tol))
                break
        x = np.where(x < nu, 0, x)
    else:                               # pragma: no cover
        # Standard Newton's method
        def newton(x):
            return x - J_(x)/dJ_(x)

        x0 = x
        x = newton(x0)
        while np.max(abs((x - x0)/x)) > rel_tol:
            x0 = x
            x = newton(x0)

    return x


def j_root(nu, N, rel_tol=2*_EPS):
    r"""Return the first N positive roots of the bessel function
    `J_nu(x)`.

    Parameters
    ----------
    nu : float
       Order of bessel function
    N : int
       Number of roots
    rel_tol : float, optional
       Desired relative tolerance for roots.

    Notes
    -----
    The general method is to first estimate the roots with a
    bisection/secant method, and then polish them using Newton's
    method.

    We start by estimating the lower bound for the first for
    non-negative :math:`\nu`

    .. math::
       \nu + \nu^{1/3} < j_{\nu,1} < \nu + 1.85575 \nu^{1/3} + \pi

    Then, using the fact that the roots are spaced by at least
    :math:`\pi`, we step through the sign changes to bracket all of
    the desired roots.

    lowest root using the following
    heuristics

    .. math::
       j_{\nu,s} \approx \begin{cases}
          2\sqrt{\nu + 1}
             & -1 < \nu < -0.8\\
          \left(\frac{\nu}{2} + \frac{3}{4}\right)\pi
             & -0.8 < \nu < 2.5\\
          \nu + 1.85575 \nu^{1/3}
             & 2.5 \leq \nu
       \end{cases}

    Examples
    --------
    >>> nu = 2.5
    >>> j_ = j_root(nu, 2000)
    >>> J_ = J(nu)(j_)

    These are roots!

    >>> np.max(abs(J_/j_)) < _EPS
    True

    They are also distinct

    >>> pi < min(np.diff(j_))
    True

    And the spacing is decreasing, meaning we have not skipped any.

    >>> np.max(np.diff(np.diff(j_))) < 0
    True
    """
    J_ = J(nu)
    dJ_ = J(nu, 1)

    nu2 = 2*nu

    if nu2 < 0:                         # pragma: no cover
        raise ValueError("mu must be non-negative")
    elif 1 == nu2:
        # Roots of sin(x)/x = 0:
        # x = pi*n excluding n=0
        return pi*np.arange(1, N+1)
    elif 3 == nu2:
        # Roots of sin(x)/x**2 - cos(x)/x:
        # x = tan(x) excluding x = 0
        # If n > 10 iterate x :-> n*pi + arctan(x)
        # 5 times starting with x = pi*(n+0.5)
        x = np.array([4.4934094579090642, 7.7252518369377068,
                      10.904121659428899, 14.066193912831473,
                      17.22075527193077,  20.371302959287561,
                      23.519452498689006, 26.666054258812672,
                      29.811598790892958, 32.956389039822476])
        if N > 10:
            n = np.arange(11, N+1)
            npi = n*pi
            x0 = (n+0.5)*pi
            for c in xrange(5):
                np.arctan(x0, x0)
                x0 += npi
            return np.hstack((x, x0))
        else:
            return x[:N]
    else:
        # Find brackets.
        x = np.empty(N+1, dtype=float)
        Jx = np.empty(N+1, dtype=float)

        x[0] = nu + nu**(1./3.)
        Jx[0] = J_(x[0])
        for n in xrange(1, N+1):
            x[n] = x[n-1] + pi
            Jx[n] = J_(x[n])
            while Jx[n]*Jx[n-1] > 0:
                x[n] += pi
                Jx[n] = J_(x[n])

        # Two steps of bisection method
        x0 = x[:-1]
        x1 = x[1:]
        J0 = Jx[:-1]
        J1 = Jx[1:]
        for n in xrange(2):
            # Invariant:
            # J0*J1 < 0 or J0 = J1 = 0 and x0 = x1
            x_mid = (x0 + x1)/2
            J_mid = J_(x_mid)
            s0 = J_mid*J0
            s1 = J_mid*J1
            assert np.all(s0*s1 <= 0)
            x0 = np.where(s0 >= 0, x_mid, x0)
            x1 = np.where(s1 >= 0, x_mid, x1)
            J0 = np.where(s0 >= 0, J_mid, J0)
            J1 = np.where(s1 >= 0, J_mid, J1)
            # s0, s1 > 0 or s0 , s1 < 0: Can't happen
            # s0 < 0, s1 >= 0: J0*J1 = J0*J_mid = s0 < 0
            # s0 >= 0, s1 < 0: J0*J1 = J_mid*J1 = s1 <= 0
            # s0 = s1 = 0: x0 = x1 = x_mid and J_mid = 0

        # Now form guess using secant method.
        x = (J1*x0 - J0*x1)/(J1 - J0)
        return j_root_x(nu=nu, x=x, rel_tol=rel_tol)


def J_sqrt_pole(nu, zn, n=0):
    r"""Return a function that computes the `n`'th derivative of
    `sqrt(z)*J(nu,z)/(z - zn)` where `zn` is a root: `J(nu, zn) = 0`.

    Parameters
    ----------
    nu : float
       Order
    zn : float
       Root of `J(nu, z)`
    n : int
       Order of derivative to take.

    Notes
    -----
    .. math::
       \frac{\sqrt{z}J_{\nu}(z)}{z - z_{n}}

    As :math:`z` approaches :math:`z_n`, this has the form of `0/0`,
    so one can apply a form of l'Hopital's rule to reduce the
    round-off error.  The specified form of the function has been
    chosen for special properties of the Bessel functions.  Express
    the function as

    .. math::
       F(z) &= \frac{f(z)}{z - z_n}  = \frac{\sqrt{z}J_{\nu}(z)}{z - z_n}\\
       F'(z) &= \frac{f'(z)}{z - z_n} - \frac{f(z)}{(z - z_n)^2}


    Let :math:`\delta = z - z_n`.  Close to the singular point we use
    the Taylor series:

    .. math::
       \sum_{m=0}^{\infty}\frac{a_m\delta^{m}}{m!}

    .. math::
       F(z) &= f'(z_n)
            + \sum_{m=3}^{\infty}\frac{f^{(m)}(z_n)\delta^{m-1}}{m!}
            = \sum_{m=0}^{\infty}\frac{f^{(m+1)}(z_n)\delta^{m}}{(m+1)m!}\\
        a_m &= \frac{f^{(m+1)}}{m+1}\\
       F'(z) &= \sum_{m=3}^{\infty}\frac{(m-1)f^{(m)}(z_n)\delta^{m-2}}{m!}
             = \sum_{m=1}^{\infty}\frac{f^{(m+2)}(z_n)\delta^{m}}{(m+2)m!}\\
         a_m &= \frac{f^{(m+2)}}{m+2}\\

    The first few derivatives are presented here:

    .. math::
       f(z) &= \sqrt{z}J_{\nu}(z)\\
       f'(z) &= \frac{J_{\nu}(z)}{2\sqrt{z}} + \sqrt{z}J'_{\nu}(z)
              = \frac{f(z)}{2z} + \sqrt{z}J'_{\nu}(z)\\
       f''(z) &= \sqrt{z}J_{\nu}(z)\left(
                     \frac{\nu^2 - \tfrac{1}{4}}{z^2} - 1\right)
               = f(z)\left(\frac{\nu^2 - \tfrac{1}{4}}{z^2} - 1\right)\\
       f'''(z) &= f'(z)\left(\frac{\nu^2 - \tfrac{1}{4}}{z^2} - 1\right)
               - 2f(z)\frac{\nu^2 - \tfrac{1}{4}}{z^3}\\
       f^{(4)}(z) &=
       f(z)\left[
          \left(\frac{\nu^2 - \tfrac{1}{4}}{z^2} - 1\right)^2
          + 6\frac{\nu^2 - \tfrac{1}{4}}{z^4}\right]
       - 4f'(z)\frac{\nu^2 - \tfrac{1}{4}}{z^3}
    .. Checked with Maple.

    Evaluated at the root :math:`z=z_n` these become:

    .. math::
       f(z_{n}) &= 0\\
       f'(z_{n}) &= \sqrt{z_{n}}J'_{\nu}(z_{n})\\
       f''(z_{n}) &= 0\\
       f'''(z_{n}) &= f'(z_{n})\left(
                        \frac{\nu^2 - \tfrac{1}{4}}{z_{n}^2} - 1\right)\\
       f^{(4)}(z_{n}) &= - 4f'(z_{n})\frac{\nu^2 - \tfrac{1}{4}}{z_{n}^3}

    with both the function and the second derivative vanishing.

    To determine where to use this formula, we match the estimate
    roundoff error with the truncation error.  The Bessel functions
    are of order unity and are typically calculated to an absolute
    accuracy of :math:`\epsilon`.  The round-off error in the
    numerator is :math:`\epsilon f(z)` and :math:`\epsilon \sqrt{2}
    z_n` in the denominator.  The roundoff errors in the denominator
    dominate both cases:

    .. math::
       \delta F(z) &\sim \epsilon \frac{\sqrt{2}z_n F(z)}{\delta}
                    \sim \frac{\sqrt{2}\epsilon z_n f(z)}{\delta^2}
                    \sim \frac{\sqrt{2}\epsilon z_n f'(z_n)}{\delta}\\
       \delta F'(z) &\sim \frac{2\epsilon z_n f(z)}{\delta^3}
                     \sim \frac{2\epsilon z_n f'(z_n)}{\delta^2}

    To choose the appropriate transition point, we equate half of this
    with the truncation error to transition points:

    .. math::
       \delta_c &\sim \left(
          \frac{72\epsilon z_n f'(z_n)}{\sqrt{2}f^{(4)}(z_n)}
          \right)^{1/4}
       \sim \left(\frac{72\epsilon z_n}{\sqrt{2}} \right)^{1/4}\\
       \delta_c' &\sim \left(120\epsilon z_n\right)^{1/5}

    the fact that :\math:`f(z)` behaves
    asymptotically as a :math:`\sqrt{2/\pi}\cos(z + \phi)` and so all
    derivatives have essentially the same magnitude.

    Examples
    --------
    >>> nu = 5.5
    >>> zn = j_root(nu,21)[-1]
    >>> abs(zn - 73.62361318251753391646) < 1e-16
    True
    >>> float(J_sqrt_pole(nu,zn)(zn))   # doctest: +ELLIPSIS
    -0.796778576780013...

    -0.796778576780013129760

    You can also use a vector of `zn`, but only if it is commensurate
    with the argument:

    >>> zn = j_root(nu,21)
    >>> float(J_sqrt_pole(nu, zn)(zn[-1])[20]) # doctest: +ELLIPSIS
    -0.796778576780013...
    """
    J_ = J(nu)
    dJ = J(nu, 1)

    # Taylor coefficients
    c = (nu*nu - 0.25)/zn/zn

    fzn = np.zeros(7, dtype=object)
    fzn[1] = np.sqrt(zn)*dJ(zn)
    fzn[3] = (c - 1)*fzn[1]
    fzn[4] = -4*c/zn*fzn[1]
    fzn[5] = (18*c/zn/zn + (c-1)**2)*fzn[1]
    fzn[6] = -12*(8/zn/zn + (c-1))*c/zn*fzn[1]

    m = np.arange(0, len(fzn) - 1)
    a_F = fzn[m+1]/(m+1)

    m = np.arange(0, len(fzn)-2)
    a_dF = fzn[m+2]/(m+2)

    # A more complicated estimate could be made here, but one must be
    # careful about cases such as nu = 0.5 where coefficients vanish.
    f1_f6 = 1.0  # fzn[1]/fzn[6]

    delta_c = np.abs(720*np.sqrt(2)*_EPS*zn*f1_f6)**(1/6)
    ddelta_c = np.abs(144*2*_EPS*zn*f1_f6)**(1/6)

    def f(z, J=J_):
        return np.sqrt(z)*J(z)

    def df(z, J=J_, dJ=dJ):
        return J(z)/2/np.sqrt(z) + np.sqrt(z)*dJ(z)

    if 0 == n:
        def F(z, zn=zn, delta_c=delta_c, f=f, a_F=a_F):
            denom = z - zn
            return np.where(abs(denom) > delta_c,
                            np.divide(f(z), denom + _TINY),
                            _Horner(a_F, denom))
        return F
    elif 1 == n:
        def dF(z, zn=zn, ddelta_c=ddelta_c, f=f, df=df, a_dF=a_dF):
            denom = z - zn
            return np.where(abs(denom) > ddelta_c,
                            np.divide(df(z) - np.divide(f(z), denom),
                                      denom),
                            _Horner(a_dF, denom))
        return dF
    else:                               # pragma: no cover
        raise ValueError("Only n=0 or 1 supported.")


def _Horner(a, d):
    """Return sum(a[n]/n!*d^n) evaluated using Horner's
    method.

    Examples
    --------
    >>> a = [1, 1, 2, 3*2, 4*3*2]
    >>> d = 2
    >>> _Horner(a,d)
    31.0
    """
    d = np.asarray(d)
    ans = 0*d
    for n in reversed(xrange(len(a))):
        ans += a[n]
        if n > 0:
            ans *= d/n
    return ans
