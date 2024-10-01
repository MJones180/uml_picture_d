# Have NumPy run on a single core.
import single_core_numpy.py  # noqa: F401
from main import main

# main_scnp = main - single core NumPy
if __name__ == '__main__':
    # Call the `main` script like normal
    main()
