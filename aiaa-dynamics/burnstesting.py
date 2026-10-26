from astropy.time import Time

from matplotlib import pyplot as plt

from hapsira.bodies import Earth, Mars, Jupiter, Sun
from hapsira.frames import Planes
from hapsira.plotting import OrbitPlotter
from hapsira.plotting.orbit.backends import Matplotlib2D
from hapsira.twobody import Orbit
epoch = Time("2018-08-17 12:05:50", scale="tdb")

plotter = OrbitPlotter(plane=Planes.EARTH_ECLIPTIC)
plotter.plot_body_orbit(Earth, epoch, label="Earth")
plotter.plot_body_orbit(Mars, epoch, label="Mars")
plotter.plot_body_orbit(Jupiter, epoch, label="Jupiter")


