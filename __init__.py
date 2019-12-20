from .interface import *
def backend(name):
    if name == 'gorn':
        from . import backend_gorn
        backend_gorn.init()

#     if name == 'lotov':
#         from . import backend_lotov
#         backend_lotov.init()

#     if name == 'minakov':
#         from . import backend_minakov
#         backend_minakov.init()
