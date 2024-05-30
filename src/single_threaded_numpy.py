"""
Flags to make NumPy run on a single thread.
"""

from os import environ

# NumPy
environ['MKL_NUM_THREADS'] = '1'
environ['NUMEXPR_NUM_THREADS'] = '1'
environ['OMP_NUM_THREADS'] = '1'
environ['OPENBLAS_NUM_THREADS'] = '1'
