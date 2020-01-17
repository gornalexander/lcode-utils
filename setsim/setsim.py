from ..interface import *
from .config import *
from ..backend_gorn.beam_gen import Gauss, rGauss
import os
from shutil import copyfile
import matplotlib.pyplot as plt


def printmd(string):
    try:
        from IPython.display import Markdown, display
        display(Markdown(string))
    except:
        print('### ', string, ' ###')

#-------------------------
# Making directory module
#-------------------------
def makedir(path, name, cfg=config): 
    path = os.path.join(path, name)
    print('Making directory... ', end='')
    try:
        os.makedirs(path)
        print('done.')
    except:
        print('seems like the directory already exists.')
    
    with open(os.path.join(path, 'lcode.cfg'), "w") as text_file:
        print('Writing lcode.cfg... ', end='')
        text_file.write(cfg)
    print('done')
    return path

#------------------------------
# Simulation setting up module
#------------------------------
def setup_simulation(lcode, time_step, window_width, window_length):
    lcode.set_parameter('time-step', time_step)
    print(type(window_width))
    window_width = round(window_width, 2)
    lcode.set_parameter('window-width', window_width)
    window_length = float(window_length)
    lcode.set_parameter('window-length', window_length)
    print('')

#----------------------
# Making plasma module
#----------------------
def plasma_zshape(x, n, form='L'):
    zshape = ''
    dxs = x[1:] - x[:-1]
    arr = np.array([dxs, n[:-1], n[1:]]).T
    zshape += 'plasma-zshape = """\n'
    for dx, n_start, n_end in arr:
        zshape += '%f %f %s %f\n' % (dx, n_start, form, n_end)
    zshape +='"""'
    return zshape

def make_plasma(lcode, n_plasma, plasma_length, plasma_radius, n_profile):
    print("# PLASMA")
    plasma_length = round(plasma_length)
    lcode.set_parameter('time-limit', plasma_length + 0.1)

    plasma_radius = round(plasma_radius, 2)
    lcode.set_parameter('plasma-width', plasma_radius)

    # z-shape
    n0 = n_plasma
    z = np.arange(0, plasma_length, lcode.get_parameter('time-step'))
    nz = np.array([n0*n_profile(x) for x in z])
    plt.title('density profile')
    plt.plot(z*lcode.cwp/100, lcode.n*nz)
    plt.xlabel(r'$z$ [m]')
    plt.ylabel(r'$n$ [cm-3]')
    print('Writing pzshape.txt... ', end='')
    with open(os.path.join(lcode.path, 'pzshape.txt'), 'w') as f:
        f.write(plasma_zshape(z, nz))
    print('done\n')

#------------------------------
# Beamfile construction module
#------------------------------
def Q2I(Q, sigma_z): # SI (returns Amps)
    return c/100. * Q / np.sqrt(2*np.pi) / sigma_z
def make_beam(lcode, sigma_z, pos_xi, sigma_r, p0, enspread, emittance, N_particles, q_m=m_MeV/M_MeV, q_e=1):
    print("# BEAM")
    # xi
    xi_shape = Gauss(pos_xi, sigma_z)
    # r
    r_shape = rGauss(sigma_r)
    # pz
    pz_shape = Gauss(p0, enspread)
    # angle
    angspread = emittance / (p0 * q_m * sigma_r)
    #print('%e' % angspread)
    ang_shape = Gauss(0, angspread)
    # current
    Ipeak_A = Q2I(N_particles*e*q_e/3e9, sigma_z*lcode.cwp/100)
    Ipeak_kA = Ipeak_A/1000
    print('Peak current {:.2f} A'.format(Ipeak_A))
    lcode.beam = lcode.make_beam(xi_shape, r_shape, pz_shape, ang_shape, Ipeak_kA, q_m, saveto=lcode.path)
    print('Writing beamfile.bit...', end='')
    with open(os.path.join(lcode.path, 'beamfile.bit'), 'w') as f:
        f.write('0.0')
    print('done.\n')

#--------------------
# Making task module
#--------------------
def make_task(lcode, name, N_nodes, cluster='matrosov'):
    if cluster == 'matrosov':
        task = """#!/bin/bash
#
#PBS -N {}
#PBS -l nodes={}:intel:ppn=36,pvmem=30000mb,walltime=200:00:00
cd $PBS_O_WORKDIR
/share/apps/bin/mpiexec -perhost 36 lcode lcode.cfg pzshape.txt""".format(name, N_nodes)
        print('Making task.sh... ', end='')
        with open(os.path.join(lcode.path, 'task.sh'), 'w') as f:
            f.write(task)
        print('done.\n')
    elif cluster == 'matrosov2':
        task = """#!/bin/bash
#
#PBS -N {}
#PBS -l nodes={}:amd:ppn=32,pvmem=30000mb,walltime=200:00:00
cd $PBS_O_WORKDIR
/share/apps/bin/mpiexec -perhost 32 lcode2 lcode.cfg pzshape.txt""".format(name, N_nodes)
        print('Making task.sh... ', end='')
        with open(os.path.join(lcode.path, 'task.sh'), 'w') as f:
            f.write(task)
        print('done.\n')
    elif cluster == 'condor':
        print('Making condor directory... ', end='')
        try:
            os.makedirs(os.path.join(lcode.path, 'condor'))
            print('done.')
        except:
            print('seems like the directory already exists.')
        print('Copying condor files... ', end='')
        copyfile('condor/job.py', os.path.join(lcode.path, 'condor'))
        copyfile('condor/job_start.sh', os.path.join(lcode.path, 'condor/job_%s.sh' % name))
        print('done.\n')