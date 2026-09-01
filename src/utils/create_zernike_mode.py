"""
Code to generate Zernike modes. This script was created with Claude.
"""
import numpy as np
import math


def noll_to_nm(j):
    n = int(np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2))
    j_curr = n * (n + 1) // 2 + 1
    for m_abs in range(n % 2, n + 1, 2):
        if m_abs == 0:
            if j_curr == j:
                return n, 0
            j_curr += 1
        else:
            if j_curr % 2 == 0:
                m_first, m_second = m_abs, -m_abs
            else:
                m_first, m_second = -m_abs, m_abs
            if j_curr == j:
                return n, m_first
            if j_curr + 1 == j:
                return n, m_second
            j_curr += 2


def create_zernike_mode(j, grid_size):
    n, m = noll_to_nm(j)
    X, Y = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size),
    )
    m_abs = abs(m)
    rho2 = X**2 + Y**2

    # Step 1: Compute the radial polynomial divided by rho^|m|
    # R_n^|m|(rho) = rho^|m| * sum_k C_k * rho^(n - |m| - 2k)
    # The sum part is a polynomial in rho^2 alone:
    # Q(rho^2) = sum_k C_k * (rho^2)^((n - |m|)/2 - k)
    Q = np.zeros_like(X, dtype=float)
    for k in range((n - m_abs) // 2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = (math.factorial(k) * math.factorial((n + m_abs) // 2 - k) *
               math.factorial((n - m_abs) // 2 - k))
        power = (n - m_abs) // 2 - k  # guaranteed non-negative integer
        Q += (num / den) * (rho2**power)

    # Step 2: Compute (X + iY)^|m| — a pure polynomial in X and Y
    # Then take real part for cos modes, imaginary part for sine modes
    # This encodes rho^|m| * cos(|m|*theta) and rho^|m| * sin(|m|*theta)
    # as exact polynomials with no singularity anywhere
    if m_abs == 0:
        angular = np.ones_like(X, dtype=float)
    else:
        # Expand (X + iY)^m_abs using binomial theorem
        # = sum_{p=0}^{m_abs} C(m_abs, p) * X^(m_abs-p) * (iY)^p
        real_part = np.zeros_like(X, dtype=float)
        imag_part = np.zeros_like(X, dtype=float)
        for p in range(m_abs + 1):
            coeff = math.comb(m_abs, p)
            # (iY)^p = i^p * Y^p
            # i^0=1, i^1=i, i^2=-1, i^3=-i, repeating
            i_power = p % 4
            x_part = X**(m_abs - p) * Y**p * coeff
            if i_power == 0:
                real_part += x_part
            elif i_power == 1:
                imag_part += x_part
            elif i_power == 2:
                real_part -= x_part
            elif i_power == 3:
                imag_part -= x_part
        if m >= 0:
            angular = real_part  # cos(m*theta) * rho^m
        else:
            angular = imag_part  # sin(|m|*theta) * rho^m

    # Step 3: Full Zernike = Q(rho^2) * angular_polynomial(X, Y)
    # Both factors are pure polynomials — product is smooth everywhere
    zernike = Q * angular

    # Step 4: Noll normalization
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    zernike = zernike * norm

    return zernike
