import numpy as np
import density as dens

Ni = 256
L = 25.
m = (L/Ni) ** 2.

# Load N-body simulation
positions_initial = np.fromfile('data/positions_initial.bin',  dtype='float64').reshape((-1,2))
positions  = np.fromfile('data/positions.bin',  dtype='float64').reshape((-1,2))
velocities = np.fromfile('data/velocities.bin', dtype='float64').reshape((-1,2))

# Range of the density field
rangeX = np.arange(-0.05 * L, 0.05 * L, L / (8. * Ni))
grid = np.array(np.meshgrid(rangeX, rangeX)).transpose(1,2,0).reshape((-1,2))

# DTFE
print('')
print('Delaunay Tesselation Field Estimator')
dtfe = dens.DTFE(positions, velocities, m)
print('rho:', dtfe.density([0., 0.]), 
      'v:', dtfe.v([0., 0.]))
density = np.array(list(map(lambda p: dtfe.density(p), grid)))
density.astype('float64').tofile('data/density_DTFE.bin')

vel = np.array(list(map(lambda p: dtfe.v(p), grid)))
vel.astype('float64').tofile('data/v_dtfe.bin')

# Phase-Space Estimator
print('')
print('Phase-Space Estimator')
ps = dens.PhaseSpace(positions_initial, positions, m, depth = 10, box = np.array([[-L, L], [-L, L]]))
print('rho:', ps.density([0., 0.]),
      'streams:', ps.numberOfStreams([0., 0.]))
density = np.array(list(map(lambda p: ps.density(p), grid)))
density.astype('float64').tofile('data/density_PS.bin')

# Phase-Space DTFE
print('')
print('Phase-Space Delaunay Tesselation Field Estimator')
ps_dtfe = dens.PhaseSpaceDTFE(positions_initial, positions, velocities, m, depth = 10, box = np.array([[-L, L], [-L, L]]))
print('rho:', ps_dtfe.density([0., 0.]), 
      'streams:', ps_dtfe.numberOfStreams([0., 0.]), 
      'v:', ps_dtfe.v([0., 0.]))
density = np.array(list(map(lambda p: ps_dtfe.density(p), grid)))
density.astype('float64').tofile('data/density_PS-DTFE.bin')

streams = np.array(list(map(lambda p: ps_dtfe.numberOfStreams(p), grid)))
streams.astype('float64').tofile('data/streams.bin')
