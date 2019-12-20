config = """
# Simulation area:
geometry = c
window-width = 10;		r-step = 0.01
window-length = 1500;		xi-step = 0.01
time-limit = 50000.1;		time-step = 200

# Particle beams:
beam-current = 0.00281
rigid-beam = n
beam-substepping-energy = 810000
beam-particles-in-layer = 200
beam-profile = \"\"\"
xishape=b, ampl=1.0, length=1500, rshape=g, radius=1.0, angshape=l, angspread=4.5e-5, energy=7.83e5, m/q=1836, eshape=g, espread=2740.0
\"\"\"

# Plasma:
plasma-model = P # Plasma model (fluid/particles/newparticles, f/p/P)
plasma-particles-number = 5000
plasma-profile = 2 # Initial profile (uniform/stepwise/gaussian/arbitrary/channel)
plasma-width = 2
plasma-temperature = 0
ion-model = Y # Model of plasma ions (mobile/background/absent/equilibrium, Y/y/n/N)
ion-mass = 157000
substepping-depth = 3
substepping-sensivity = 0.2

# Every-time-step diagnostics:
indication-line-format = 1 # On-screen indication line format (eacht/eachxi)
output-Ez-minmax = y;		output-Ez-local = y
output-Phi-minmax = y;		output-Phi-local = y
output-lost-particles = y

# Periodical diagnostics:
output-time-period = 5000

#  Colored maps: (Er,Ef,Ez,Phi,Bf,Bz,pr,pf,pz,pri,pfi,pzi
#                 nb,ne,ni,Wf,dW,SEB,Sf,Sf2,Sr,Sr2,dS,dS2):
colormaps-full = ""
colormaps-subwindow = ""
colormaps-type = F
drawn-portion = 1 # Drawn portion of the simulation window
subwindow-xi-from = -0;		subwindow-xi-to = -100
subwindow-r-from = 0;		subwindow-r-to = 10
output-reference-energy = 1000
output-merging-r = 2;		output-merging-z = 50
output-skipping-r = 20;          output-skipping-xi = 20
palette = d # Colormaps palette (default/greyscale/hue/bluewhitered, d/g/h/b)
                E-step = 0.059;	               nb-step = 0.00056
              Phi-step = 0.059;	               ne-step = 0.1
               Bf-step = 0.059;	               ni-step = 0.01
               Bz-step = 0.059;	             flux-step = 0.02
 electron-momenta-step = 0.1;	 r-corrected-flux-step = 0.02
      ion-momenta-step = 0.1;	           energy-step = 10

#  Output of various quantities as functions of xi:
#   (ne,nb,Ez,<Ez>,Bz,Phi,pz,emitt,dW,Wf,ni,pzi)
#   (nb2,Er,Ez2,Bf,Bz2,Fr,pr,pf,<rb>,dS,Sf,SEB,pri,pfi,Ef)
f(xi) = Ez,Ez2,Er,Bf
f(xi)-type = Y
axis-radius = 0;		auxillary-radius = 1
               E-scale = 0.59;	              nb-scale = 0.02
             Phi-scale = 0.59;	              ne-scale = 2
              Bz-scale = 0.59;	              ni-scale = 0.1
electron-momenta-scale = 0.5;	            flux-scale = 0.5
     ion-momenta-scale = 0.5;	          energy-scale = 1
     beam-radius-scale = 5;	       emittance-scale = 300

#  Beam particle information as pictures (r,pr,pz,M):
output-beam-particles = ""
draw-each = 1
beam-picture-height = 900
beam-pr-scale = 1000
beam-a-m-scale = 1000;		beam-pz-scale = 15000

# Output of beam characteristics in histogram form (r,z,M,a):
histogram-output = ""
histogram-output-accel = ""
histogram-type = y
histogram-bins = 300;		beam-angle-scale = 0.02

#  Trajectories of plasma particles:
trajectories-draw = n
trajectories-each = 50;	trajectories-spacing = 100
trajectories-min-energy = 1;	trajectories-energy-step = 0.5

# Saving run state periodically:
saving-period = 50000
save-beam = y
save-plasma = n
"""