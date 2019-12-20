#!/usr/bin/env python3

#---------
# Imports
#---------
from .units import * #SGC is a default unit system
from shutil import copyfile
import sys, os
import pandas as pd
from pandas import DataFrame  as df
import numpy as np
from numpy import sqrt, pi
from random import randrange
import scipy.stats as stats
import re
import h5py



#---------------
# Readind utils
#---------------
def find(cfg, par):
    ans = re.search('\s' + par + '\s?=\s?[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', cfg)
    return float(ans.group(0).replace(par,'').replace('=', ''))

def find_char(cfg, par):
    ans = re.search(par + '\s?=\s?[a-z]', cfg)
    return ans.group(0).replace(par,'').replace('=', '').replace(' ','')


#-----------------
# Interface class
#-----------------

class Interface():
    backend = None
    # INITIALIZATION
    def __init__(self, path = '', read = None, n = 7e14):
        if self.backend is None:
            raise Exception('Choose backend first')
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
    
    # SETTING AND GETTING A PARAMETER IN lcode.cfg
    def set_parameter(self, par, val):
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
        with open(os.path.join(self.path, 'lcode.cfg'), 'r') as config_file:
            config = config_file.read()
        return find(config, par)
    
    # SETTING AND CHANGING PLASMA DENSITY
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

    # WORKING WITH BEAMFILES
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
    def calculate_xy(self):
        # first calculate random angle:
        self.beam['phi'] = 2*np.pi*np.random.random(size=len(self.beam))
        self.beam['x'] = self.beam['r']*np.cos(self.beam['phi'])
        self.beam['y'] = self.beam['r']*np.sin(self.beam['phi'])
        self.beam['pf'] = self.beam.M/(self.beam.r + 1e-15)
        self.beam['px'] = self.beam['pr']*np.cos(self.beam['phi']) - self.beam['pf']*np.sin(self.beam['phi'])
        self.beam['py'] = self.beam['pr']*np.sin(self.beam['phi']) + self.beam['pf']*np.cos(self.beam['phi'])
        return False
    def add_beam(self, name='beamfile.bin', tofile=False):
        try:
            if self.beam is None:
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


    # WORKING WITH FIELDS
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

