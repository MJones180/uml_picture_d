import sys
from utils.constants import PACKAGES_P
from utils.path import get_abs_path

# Look for modules in the `packages` directory
packages_path = str(get_abs_path(PACKAGES_P))
print('Adding to paths: ', packages_path)
sys.path.append(packages_path)
