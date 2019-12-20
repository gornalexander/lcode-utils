import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
from ..units import *

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

def plot_beam(self, x='xi', y='pz' , cond=None, x_units='', y_units='', hist=True, **kwargs):
    columns = {'xi': 0, 'r': 0, 'x': 0, 'pz': 1, 'pr': 1, 'pf': 1, 'M': 2, 'q_m': 2, 'q': 2, 'N': 2}
    tolatex = {'xi': r'\xi', 'r': 'r', 'x': 'x', 'pz': 'p_z', 'pr': 'p_r', 'pf': r'p_\phi', 'M': 'M', 'q_m': 'q/m', 'q': 'q', 'N': 'N'}
    if (x, y not in columns.keys())[-1]:
        print('Choose x, y from:', columns.keys())
        return True

    if self.beam is None and self.add_beam():
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
    plt.grid(alpha=0.5)
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

def plot_emaxf(self, y='emax', x_units='m', y_units='MV/m', **kwargs):
        tolatex = {'emax': 'E_{max}', 'emin': 'E_{min}', 'xi_max': '\\xi_{max}', 'xi_min': '\\xi_{min}'}
        column = {'emax': 1, 'emin': 3, 'xi_max': 2, 'xi_min': 4}
        units = {'nm': 1e7*c/self.wp, 'um': 1e4*c/self.wp, 'mm': 10*c/self.wp, 'cm': 1*c/self.wp,
                 'm': 1e-2*c/self.wp, 'c/wp': 1., 'ps': 1e+12/self.wp, 'V/m': self.E0_MVm*1e6,
                 'kV/m': self.E0_MVm*1e3, 'MV/m': self.E0_MVm,
                 'GV/m': self.E0_MVm/1e3, 'TV/m': self.E0_MVm/1e6, '': 1.}
        try:
            emaxf = np.loadtxt(os.path.join(self.path, 'emaxf.dat')).T
            zlim = self.get_parameter('time-limit')     
        except:
            print('There is no emaxf.dat or lcode.cfg')
            return True
        
        N_steps = zlim // self.t_step
        zlim = N_steps * self.t_step
        plt.plot(emaxf[0] * units[x_units], emaxf[column[y]] * units[y_units], 'r', **kwargs)
        
        plt.xlim(0, zlim * units[x_units])
        plt.xlabel('$z$ ({})'.format(x_units))
        plt.ylabel('${}$ ({})'.format(tolatex[y], y_units))
        return False