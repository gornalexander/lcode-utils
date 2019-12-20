import scipy.stats as stats
import os
import pandas as pd
from pandas import DataFrame  as df
import numpy as np
from numpy import sqrt, pi


# BEAM PROFILES
def Uniform(start=0, length=1):
    def uniform_maker(N):
        return stats.uniform.ppf(np.linspace(0, 1, N+2)[1:-1], start-length, length)
    uniform_maker.med = start - length/2.
    uniform_maker.sigma = length
    uniform_maker.f0 = 1. / length 
    return uniform_maker

def rUniform(width=1):
    def runiform_maker(N):
        return np.random.triangular(0, width, width, N)
    runiform_maker.med = 0
    runiform_maker.sigma = width
    return runiform_maker

def Gauss(med=0, sigma=1):
    def gauss_maker(N):
        return stats.norm.ppf(np.linspace(0, 1, N+2)[1:-1], med, sigma)
    gauss_maker.med = med
    gauss_maker.sigma = sigma
    gauss_maker.f0 = 1. / sqrt(2*pi) / sigma
    return gauss_maker

def rGauss(sigma=1):
    def rgauss_maker(N):
        return sigma*np.sqrt(2.)*stats.weibull_min.ppf(np.linspace(0, 1, N+2)[1:-1], 2)
    rgauss_maker.med = 0
    rgauss_maker.sigma = sigma
    return rgauss_maker

def make_beam(self, xi_distr, r_distr, pz_distr, ang_distr, Ipeak_kA, q_m=1.0, partic_in_layer=200, savehead=False, saveto='./', name='beamfile.bin'):
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
        I0 = 17.03331478052319 # kA
        q = 2.*Ipeak_kA/I0/partic_in_layer
        stub_particle = np.array([[-100000., 0., 0., 0., 0., 1.0, 0., 0.]])
        gamma = pz_distr.med 
        N = partic_in_layer / self.xi_step / xi_distr.f0
        N = int(round(N))
        xi = xi_distr(N)
        if savehead:
            xi = xi[xi >= -self.xi_size]
            N = len(xi)
        else:
            xi = xi[(xi >= -self.xi_size) & (xi <= 0)]
            N = len(xi)
        partic_in_mid_layer = np.sum((xi > xi_distr.med - self.xi_step/2) & (xi < xi_distr.med + self.xi_step/2))
        print('Number of particles:', N)
        print('Number of particles in the middle layer:', partic_in_mid_layer)
        xi = np.sort(xi)[::-1]
        r = np.abs(r_distr(N))
        np.random.shuffle(r)
        pz = pz_distr(N)
        np.random.shuffle(pz)
        pr = gamma * ang_distr(N)
        np.random.shuffle(pr)
        M = gamma * ang_distr(N)
        np.random.shuffle(M)
        M = M * r
        particles = np.array([xi, r, pz, pr, M, q_m * np.ones(N), q * np.ones(N), np.arange(N)])
        beam = np.vstack([particles.T, stub_particle])
        beam = df(beam, columns=['xi', 'r', 'pz', 'pr', 'M', 'q_m', 'q', 'N'])
        head = beam[beam.eval('xi>0')]
        beam = beam[beam.eval('xi<=0'.format(-self.xi_size))]
        if saveto:
            beam.values.tofile(os.path.join(saveto, name))
        if savehead:
            head.values.tofile(os.path.join(saveto, 'head-' + name))
        return beam