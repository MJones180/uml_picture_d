"""
Flags to make NumPy and its associated libraries run on a single core.
"""

from os import environ

# NumPy
environ['MKL_NUM_THREADS'] = '1'
environ['NUMEXPR_NUM_THREADS'] = '1'
environ['OMP_NUM_THREADS'] = '1'
environ['OPENBLAS_NUM_THREADS'] = '1'
