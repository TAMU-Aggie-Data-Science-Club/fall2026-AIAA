
####################################################
#             AIAA Burns Simulation                #
#         CODE WRITTEN BY AARYA SHEWALE            #
#   IF NEED HELP, CONTACT "aaryashewale@tamu.edu"  #
####################################################

###########
# IMPORTS #
###########

# Astropy: physical units and time handling
import astropy.units as units
from astropy import time
from astropy.time import Time

# NumPy: basic numerical routines
import numpy as np

# jplephem: planetary ephemerides via SPICE/SPK kernels
from jplephem.spk import SPK

# matplotlib: static orbit plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3D projection)

# numba: JIT acceleration (CPython only)
from numba import njit

# Plotly: interactive orbit plotting
import plotly.graph_objects as go

# SciPy: root finding and numerical propagation
from scipy.optimize import brentq, newton
from scipy.integrate import solve_ivp

# Haspira: Earth, Moon, Sun celesial bodies 
import hapsira as haspira


class Burns:

    def __init__(self, J )
    J = 0

   # def TransLunarEjection( J)
   # free return trajectory






