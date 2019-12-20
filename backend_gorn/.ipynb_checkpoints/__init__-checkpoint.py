from ..interface import *
from . import beam_gen
from . import plotlib

def init():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    Interface.backend = 'gorn'
    Interface.make_beam = beam_gen.make_beam
    Interface.plot_map = plotlib.plot_map
    Interface.plot_beam = plotlib.plot_beam
    Interface.plot_emaxf = plotlib.plot_emaxf
    