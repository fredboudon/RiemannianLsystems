# Care the path is taken from the directory python is launched
# (and not the directory that contains test.py)
import sys
sys.path.append("../RiemannTurtleLib")

from essai import foo

foo()
