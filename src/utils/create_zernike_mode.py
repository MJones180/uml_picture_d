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
    m_abs = abs(m)

    X, Y = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size),
    )
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    radial = np.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = (math.factorial(k) * math.factorial((n + m_abs) // 2 - k) *
               math.factorial((n - m_abs) // 2 - k))
        radial += (num / den) * rho**(n - 2 * k)

    if m > 0:
        angular = np.cos(m_abs * theta)
    elif m < 0:
        angular = np.sin(m_abs * theta)
    else:
        angular = 1.0

    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    return norm * radial * angular
