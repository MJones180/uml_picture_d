from pathlib import Path
import sys

# Look in the `packages` directory for the PROPER library
packages_path = f'{Path(__file__).parent.parent}/packages'
print('Adding to paths: ', packages_path)
sys.path.append(packages_path)
