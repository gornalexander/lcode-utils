import numpy as np
from numpy import sqrt, pi

#SGC is a default unit system
e = 4.8032e-10
c = 2.99792458e10
m_e = 9.10938356e-28
m_MeV = 0.5109989461
M = 1.672621898e-24
M_MeV = 938.2720813


GsToVm = 2.997825e4
VmToGs = 1/GsToVm
GsToMVm = GsToVm/1e6
MVmToGs = 1/GsToMVm
IToA = 17e3
AToI = 1/IToA

renormalize_units = lambda n: "nm, um, mm, cm, m = np.array([1e-7, 1e-4, 1e-1, 1, 100]) / (c/sqrt(4*pi*{}*e**2/m_e)); fs, ps, ns, us, ms, s = np.array([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1]) * sqrt(4*pi*{}*e**2/m_e); cm3, m3 = np.array([1, 1e6]) / {}".format(n, n, n)

# Lengths
nm, um, mm, cm, m = dist_units = np.array([1e-7, 1e-4, 1e-1, 1, 100]) / (c/sqrt(4*pi*7e14*e**2/m_e))

# Times
fs, ps, ns, us, ms, s = time_units = np.array([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1]) * sqrt(4*pi*7e14*e**2/m_e)

# Angles
mrad, rad = 1e-3, 1

# Densities
cm3, m3 = np.array([1, 1e6]) / 7e14

# Energies
eV, KeV, MeV, GeV, TeV = np.array([1, 1e3, 1e6, 1e9, 1e12])/1e6/m_MeV