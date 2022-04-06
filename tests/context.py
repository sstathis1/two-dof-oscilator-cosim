"""Adds the parent directory to the path in order to import the sample package"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))