#!/usr/bin/env python3

import sys, os
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['figure.figsize'] = (6.28, 4)
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
from pandas import DataFrame  as df
import numpy as np
from collections import namedtuple
from random import randrange
from numpy import sqrt, pi
import re
import h5py

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
cmTomkm = 1e4
cmTomm = 10
cmTom = 1e-2

# Distances
nm, um, mm, cm, m = dist_units = np.array([1e-7, 1e-4, 1e-1, 1, 100]) / 0.02

# Times
fs, ps, ns, us, ms, s = time_units = np.array([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1]) * c/0.02

# Angles
mrad, rad = 1e-3, 1

# Densities
cm3, m3 = np.array([1, 1e6]) / 7e14

# Energies
eV, KeV, MeV, GeV, TeV = np.array([1, 1e3, 1e6, 1e9, 1e12])/1e6/m_MeV

# # Distances
# dist_units = np.array([1e-7, 1e-4, 1e-1, 1, 100])

# # Times
# time_units = np.array([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1])

# Very very useful normalization for drawing colormaps
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# BEAM PROFILES
def Uniform(start=0, length=1):
    def uniform_maker(N):
        return np.random.uniform(start - length, start, N)
    uniform_maker.med = start - length/2.
    uniform_maker.sigma = length
    return uniform_maker

def rUniform(width=1):
    def runiform_maker(N):
        return np.random.triangular(0, width, width, N)
    runiform_maker.med = 0
    runiform_maker.sigma = width
    return runiform_maker

def Gauss(med=0, sigma=1):
    def gauss_maker(N):
        return np.random.normal(med, sigma, N)
    gauss_maker.med = med
    gauss_maker.sigma = sigma
    return gauss_maker

def rGauss(sigma=1):
    def rgauss_maker(N):
        return sigma*np.sqrt(2.)*np.random.weibull(2, N)
    rgauss_maker.med = 0
    rgauss_maker.sigma = sigma
    return rgauss_maker


# READING
def find(cfg, par):
    ans = re.search('\s' + par + '\s?=\s?[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', cfg)
    return float(ans.group(0).replace(par,'').replace('=', ''))
def find_char(cfg, par):
    ans = re.search(par + '\s?=\s?[a-z]', cfg)
    return ans.group(0).replace(par,'').replace('=', '').replace(' ','')

"""
Read:
    PARTICLES:
    beamfile.bin +
    partic.swp
    pr*.swp
    pz*.swp
    pf*.swp
    FIELDS:
    xi_Er_*.swp
    xi_Ez_*.swp
    xi_Er2_*.swp
    xi_Ez2_*.swp
    er*.swp +
    ez*.swp +
    ef*.swp +
    br*.swp +
    bz*.swp +
    bf*.swp +
"""

class LCODEplot():
    #Class plots stantart LCODE output. Specify the path to the directory, where lcode executable file is. 
    def __init__(self, path = '', read = None, n = 7e14):
        # read = [beam, partic, ...]
        self.path = path
        try:    
            config = open(os.path.join(path, 'lcode.cfg'), 'r').read()
            self.geometry = find_char(config, 'geometry')
            self.config = config
            self.r_size = find(config, 'window-width')
            self.xi_size = find(config, 'window-length')
            self.r_step = find(config, 'r-step')
            self.xi_step = find(config, 'xi-step')
            self.t_step = find(config, 'time-step')
            self.beam_partic_in_layer = find(config, 'beam-particles-in-layer')
            
        except:
            print("""There is no lcode.cfg or some parameters in it. 
            To draw field maps define: r_size, xi_size, r_step, xi_step, t_step.
            To generate a beamfile define: beam_partic_in_layer""")
        self.beam = None
        self.geometry = None
        self.F = {}
        self._n0 = n
        self.wp = sqrt(4*pi*n*e**2/m_e)
        self.E0_MVm = m_e*c*self.wp/e*GsToMVm
#         global nm, um, mm, cm, m
#         global fs, ps, ns, us, ms, s
#         nm, um, mm, cm, m = dist_units / (c/self.wp)
#         fs, ps, ns, us, ms, s = time_units * self.wp
    def set_parameter(self, par, val):
        config = ''
        with open(os.path.join(self.path, 'lcode.cfg'), 'r') as config_file:
            config = config_file.read()
        part = re.search('\s' + par + '\s?=\s?[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', config).group(0)
        mask = '{}={}' if part[0] == ' ' else '{} = {}'
        config = re.sub('\s' + par + '\s?=\s?[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', part[0] + mask.format(par, val), config) 
        with open(os.path.join(self.path, 'lcode.cfg'), 'w') as config_file:
            print('Changing %s to %f... ' % (par, val), end='') 
            config_file.write(config)
        print('done.')
        self.config = config
        return
    def get_parameter(self, par):
        config = ''
        with open(os.path.join(self.path, 'lcode.cfg'), 'r') as config_file:
            config = config_file.read()
        return find(config, par)
    @property
    def n(self):
        return self._n0
    @n.setter
    def n(self, n):
        factor = np.sqrt(n/self._n0)
        self._n0=n
        self.wp = sqrt(4*pi*n*e**2/m_e)
        self.E0_MVm = m_e*c*self.wp/e*GsToMVm
        try:
            self.xi_step *= factor
            self.r_step *= factor
            self.xi_size *= factor
            self.r_size *= factor
            for field in self.F.keys():
                if field[0] == 'e' or field[0] == 'b':
                    self.F[field] /= factor
                elif field[0] == 'n':
                    self.F[field] /= factor**2
        except:
            print('failed transition to other plasma density')
#         global nm, um, mm, cm, m
#         global fs, ps, ns, us, ms, s
#         nm, um, mm, cm, m = dist_units / (c/self.wp)
#       fs, ps, ns, us, ms, s = time_units * self.wp
    def __read_beamfile__(self, name='beamfile.bin'):
        try:
            filename = os.path.join(self.path, name)
            dt = np.dtype([('xi', float), ('r', float), ('pz', float), ('pr', float), ('M', float), ('q_m', float), ('q', float), ('N', float)])
            arr = np.fromfile(filename, dt)
            beam = df(arr, columns = ['xi', 'r', 'pz', 'pr', 'M', 'q_m', 'q', 'N'])
            return beam
        except IOError:
            print('There is no beamfile.bin')
            return True
    def add_beam(self, name='beamfile.bin', tofile=False):
        try:
            if self.beam == None:
                self.beam = self.__read_beamfile__(name)
            else:
                self.beam = pd.concat([self.beam, self.__read_beamfile__(name)])
            if tofile:
                self.beam.values.tofile(os.path.join(self.path, tofile))
            return False
        except:
            print('Failed to add the beam {}.'.format(os.path.join(self.path, name)))
            return True
    def merge_beams(self, name1, name2, tofile=False):
        try:
            beam1 = self.__read_beamfile__(name1)
            beam2 = self.__read_beamfile__(name2)
            beam = pd.concat([beam1, beam2])
            if tofile:
                beam.values.tofile(os.path.join(self.path, tofile))
            return beam
        except:
            print('Failed to merge the beams {} and {}.'.format(os.path.join(self.path, name1), os.path.join(self.path, name2)))
            return True
    def __read_field__(self, field, time, compress, ftype='m', hdf=False):
        try:
            if hdf:
                filename = '%s%05d%s.h5' % (field, time, ftype)
                with h5py.File(os.path.join(self.path, filename)) as file:
                    self.F[field] = np.array(file['data'])[::compress, ::compress]
                return False
            else:
                filename = '%s%05d%s.swp' % (field, time, ftype)
                self.F[field] = np.loadtxt(self.path + filename)[::compress, ::compress]
                return False
        except IOError:
            print('There is no file %s' % (os.path.join(path, filename)))
            return True      
    
    def make_beam(self, xi_distr, r_distr, pz_distr, ang_distr, Ipeak_kA, q_m=1.0, partic_in_layer=200, saveto='./', name='beamfile.bin'):
        """make_beam(xi_shape, r_shape, pz_shape, ang_shape, Ipeak_kA, N_partic=10000, q_m=1.0, partic_in_layer = 200, saveto='./')"""
        if q_m == 1 and Ipeak_kA > 0:
            print('Electrons must have negative current.')
            return
        print(type(xi_distr.med))
        if xi_distr.med > 0:
            print('Beam center is in xi>0.')
        try:
            partic_in_layer=self.beam_partic_in_layer
        except:
            print('Variable partic_in_layer is not found. Default value: 200.')
        try:
            xi_step = self.xi_step
            r_size = self.r_size
        except:
            xi_step = 0.01
            r_size = 10
            print('Variable xi_step or r_size is not found. Default values: xi_step = %.3f, r_size = %3f' % (xi_step, r_size))            
        if saveto and 'beamfile.bin' in os.listdir(saveto):
            print('Another beamfile.bin is found. You may delete it using the following command: "!rm %s".' % os.path.join(saveto, name))
            return
        I0 = 17 # kA
        q = 2.*Ipeak_kA/I0/partic_in_layer
        stub_particle = np.array([[-100000., 0., 0., 0., 0., 1.0, 0., 0.]])
        gamma = pz_distr.med
        N = 10000
        while True:
            xi = xi_distr(N)
            print('Trying', N, 'particles')
            xi = xi[(-self.xi_size <= xi)]# & (xi <= 0)]
            if np.sum((xi_distr.med - xi_step/2 < xi) & (xi < xi_distr.med + xi_step/2)) < partic_in_layer:
                print(N, 'is not enough:', np.sum((xi_distr.med - xi_step < xi) & (xi < xi_distr.med)))
                N *= 10
                continue
            until_middle_layer_filled = [np.cumsum((xi_distr.med - xi_step < xi) & (xi < xi_distr.med)) <= partic_in_layer]
            xi = xi[until_middle_layer_filled]
            K = xi.shape[0]
            print(K, 'is enough')
            xi = np.sort(xi)[::-1]
            r = np.abs(r_distr(K))
            pz = pz_distr(K)
            pr = gamma * ang_distr(K)
            M = gamma * ang_distr(K) * r
            particles = np.array([xi, r, pz, pr, M, q_m * np.ones(K), q * np.ones(K), np.arange(K)])
            beam = np.vstack([particles.T, stub_particle])
            break
        beam = df(beam, columns=['xi', 'r', 'pz', 'pr', 'M', 'q_m', 'q', 'N'])
        head = beam[beam.eval('xi>0')]
        beam = beam[beam.eval('xi<=0')]
        #beam.sort_values('xi', inplace=True, ascending=False)
        if saveto:
            beam.values.tofile(os.path.join(saveto, name))
            head.values.tofile(os.path.join(saveto, 'head-' + name))
        return beam
    

    def plot_beam(self, x='xi', y='pz' , cond=None, x_units='', y_units='', hist=False, **kwargs):
        columns = {'xi': 0, 'r': 0, 'x': 0, 'pz': 1, 'pr': 1, 'pf': 1, 'M': 2, 'q_m': 2, 'q': 2, 'N': 2}
        tolatex = {'xi': r'\xi', 'r': 'r', 'x': 'x', 'pz': 'p_z', 'pr': 'p_r', 'pf': r'p_\phi', 'M': 'M', 'q_m': 'q/m', 'q': 'q', 'N': 'N'}
        if (x, y not in columns.keys())[-1]:
            print('Choose x, y from:', columns.keys())
            return True
                    
        if self.beam is None and self.__read_beamfile__():
            print('Beam is not found')
            return 
        
        beam = self.beam.copy(deep=True)[:-1]
        if self.geometry == 'p':
            beam.x = beam.x - self.r_size/2.
        
        if 'pf' in [x, y]:
                beam['pf'] = beam.M*m_MeV/beam.q_m/beam.r        
        beam.pz = beam.pz*m_MeV/beam.q_m
        beam.pr = beam.pr*m_MeV/beam.q_m
        
        l_units = {'nm': 1e7*c/self.wp, 'um': 1e4*c/self.wp, 'mm': 10*c/self.wp, 'cm': 1*c/self.wp, 
                   'm': 1e-2*c/self.wp, 'c/wp': 1., 'ps': 1e+12/self.wp}
        p_units = {'eV/c': 1e6, 'keV/c': 1e3, 'MeV/c': 1., 'GeV/c': 1e-3, 'TeV/c': 1e-6}
        other_units = {'': 1.}
        units = [l_units, p_units, other_units]
        
        if x_units == '':
            xdict = units[columns[x]]
            x_units = list(xdict.keys())[list(xdict.values()).index(1.)]
        try:
            x_scale = units[columns[x]][x_units]
        except:
            print('Wrong x_units, try: ', units[columns[x]].keys())
            return True
        if y_units == '':
            ydict = units[columns[y]]
            y_units = list(ydict.keys())[list(ydict.values()).index(1.)]
        try:
            y_scale = units[columns[y]][y_units]
        except:
            print('Wrong y_units, try: ', units[columns[y]].keys())
            return 
    
        beam[x] = x_scale*beam[x]
        beam[y] = y_scale*beam[y]
        if cond is not None:
            beam = beam[beam.eval(cond)]
        
        if hist:
            image = plt.hist2d(beam[x], beam[y], **kwargs)
        else:
            image = plt.plot(beam[x], beam[y], 'o', **kwargs)
        
        x_units = x_units.replace('wp', '\omega_p')
        x_units = ('(' + x_units + ')')*bool(x_units)
        y_units = y_units.replace('wp', '\omega_p')
        y_units = ('(' + y_units + ')')*bool(y_units)
        plt.xlabel(r"$%s\ \sf{%s}$" % (tolatex[x], x_units))
        plt.ylabel(r"$%s\ \sf{%s}$" % (tolatex[y], y_units))
        plt.grid()
        return image
  
    def plot_map(self, field, time = 1, compress = 50, x_units='c/wp', y_units='c/wp', z_units='', show=False, diverging=False, cbar=True, file_type='m', hdf=False, **kwargs):
        if field not in self.F.keys() and self.__read_field__(field, time, compress, file_type, hdf):
            return True
        l_units = {'nm': 1e7*c/self.wp, 'um': 1e4*c/self.wp, 'mm': 10*c/self.wp, 'cm': 1*c/self.wp, 
                   'm': 1e-2*c/self.wp, 'c/wp': 1., 'ps': 1e+12/self.wp}
        F_units = {'V/m': self.E0_MVm*1e6, 'kV/m': self.E0_MVm*1e3, 'MV/m': self.E0_MVm, 'GV/m': self.E0_MVm/1e3,
                   'TV/m': self.E0_MVm/1e6, '': 1.}
        fi_units = {'eV/e': m_MeV*1e6, 'keV/e': m_MeV*1e3, 'MeV/e': m_MeV, '': 1.}
        n_units = {'': 1., 'cm': self._n0, 'm': self._n0*1e6}
        units = [n_units, fi_units, F_units]
        ftolatex = {'fi': '\Phi', 'er': 'E_r', 'ez': 'E_z', 'ef': 'E_\phi', 'br': 'B_r', 'bz': 'B_z', 'bf': 'B_\phi', 'ne': 'n_e',
                   'ni': 'n_i', 'er-bf': 'E_r - B_\phi'}

        def field_type(quantity):
            if quantity == 'ne' or quantity == 'ni':
                return 0
            elif quantity == 'fi':
                return 1
            else:
                return 2
        
        def unittolatex(unit):
            f_type = field_type(field)
            if f_type == 0:
                rule = {'': 'baseline\ density', 'cm': 'cm^{-3}', 'm': 'm^{-3}'}
            elif f_type == 1 and unit == '':
                return '\Phi_0'
            elif unit == '':
                return 'E_0'
            else:
                return unit
            return rule[unit]
                
        if file_type == 'w':
            xi_min, xi_max = sorted([-self.get_parameter('subwindow-xi-from'), -self.get_parameter('subwindow-xi-to')])
            r_min, r_max = sorted([self.get_parameter('subwindow-r-from'), self.get_parameter('subwindow-r-to')])
        
        else:
            xi_min, xi_max = 0, self.xi_size
            r_min, r_max = 0, self.r_size
        
        if self.geometry == 'p':  # TODO: add proper subwindow reading
            r = np.linspace(-r_max/2, r_max/2, self.F[field].shape[1]) 
        else:
            r = np.linspace(r_min, r_max, self.F[field].shape[1])        
        
        xi = np.linspace(xi_min, xi_max, self.F[field].shape[0])
        
        xi, r = l_units[x_units]*xi, l_units[y_units]*r
        
        field_factor = units[field_type(field)][z_units]
        if diverging:
            plt.pcolormesh(-xi, r, self.F[field].T*field_factor, norm = MidpointNormalize(midpoint=0.), **kwargs)
        else:
            plt.pcolormesh(-xi, r, self.F[field].T*field_factor, **kwargs)
        if cbar:
            z_units = z_units.replace('E0', 'E_0')
            plt.colorbar(label=r'$%s\ (\sf{%s})$' % (ftolatex[field], unittolatex(z_units)))
        x_units, y_units = x_units.replace('wp', '\omega_p'), y_units.replace('wp', '\omega_p')
        plt.xlabel(r'$\xi\ (\sf{%s})$' % x_units)
        plt.ylabel(r'$r\ (\sf{%s})$' % y_units)
        if show:
            plt.show()
        return False     
    
    
    
    
    
    
    
    
#     def make_beam_old(self, xi_shape, r_shape, pz_shape, ang_shape, Ipeak_kA, q_m=1.0, xi_step=0.01, partic_in_layer=200, saveto='./'):
#         """make_beam(xi_shape, r_shape, pz_shape, ang_shape, Ipeak_kA, N_partic=10000, q_m=1.0, partic_in_layer = 200, saveto='./')"""
#         if q_m == 1 and Ipeak_kA > 0:
#             print('Electrons must have negative current')
#             return
#         try:
#             partic_in_layer=self.beam_partic_in_layer
#         except:
#             print('Variable partic_in_layer is not found. Default value: 200.')
#         try:
#             xi_step=self.xi_step
#         except:
#             print('Variable xi_step is not found. Default value: 0.01')            
#         if 'beamfile.bin' in os.listdir(saveto):
#             print('Another beamfile.bin is found. You may delete it using the following command: "!rm %s".' % os.path.join(saveto, 'beamfile.bin'))
#             return
#         xi_distr, xi_args = xi_shape
#         r_distr, r_args = r_shape
#         pz_distr, pz_args = pz_shape
#         gamma = pz_args[0]
#         ang_distr, ang_args = ang_shape
#         I0 = 17 # kA
#         q = 2.*Ipeak_kA/I0/partic_in_layer
#         beam = np.array([-100000., 0., 0., 0., 0., 1.0, 0., 0.])
#         # for i in range(N_partic):
#         #     xi, r = xi_distr(*xi_args), abs(r_distr(*r_args))
#         #     pz = pz_distr(*pz_args)
#         #     pr = gamma*ang_distr(*ang_args)
#         #     M = gamma*ang_distr(*ang_args)*r
#         #     particle = (xi, r, pz, pr, M, q_m, q, i)
#         #     beam.append(particle)
#         i = 0.
#         while len(beam[(beam.T[0] > xi_args[0] - xi_step) & (beam.T[0] < xi_args[0])]) != partic_in_layer:
#             #print(len(beam[(beam.xi > xi_args[0] - xi_step) & (beam.xi < xi_args[0])].xi))
#             xi, r = xi_distr(*xi_args), abs(r_distr(*r_args))
#             pz = pz_distr(*pz_args)
#             pr = gamma*ang_distr(*ang_args)
#             M = gamma*ang_distr(*ang_args)*r
#             particle = np.array([xi, r, pz, pr, M, q_m, q, i])
#             beam = np.vstack([beam, particle])
#             i += 1    
#         beam = sorted(beam, key=lambda x: x[0], reverse=True)
#         beam = df(beam, columns=['xi', 'r', 'pz', 'pr', 'M', 'q_m', 'q', 'N'])
#         beam.values.tofile(os.path.join(saveto, 'beamfile.bin'))
#         return beam
        
    
    
def main():
    if len(sys.argv) == 2:
        LCODE = LCODEplot(sys.argv[1])







if __name__ == '__main__':
    main()