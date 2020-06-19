"""
AllocateParticles.py - Builds a uniform bin structure and saves it to BBins.data
                        using pickle
                     - Allocates a particle to each bin by introducing a torsion
                        force and minimising the energy of the particle. Saves
                        the positions of these particles to positions.data using
                        pickle.
                    - Particles are allocated in parallel.
"""
from pathos.multiprocessing import ProcessPool
import numpy as np
import pickle

import sys
sys.path.append("../")

from WeightedEnsemble import Bins
from Alanine import Build_UnifBins, AllocateParticlesInBins, bin_id


#Number of nodes to run parallel processes on
num_nodes = 40

#Building and saving bin structure
nx_bins = 20
ny_bins = 20

print('Building Bins...', flush = True)
B = Build_UnifBins(nx_bins, ny_bins)
print('Bins Built', flush = True)

Bfile = open('BBins.data','wb')
pickle.dump(B, Bfile)
Bfile.close()

#Allocating particles in parallel
num_steps = [10**3 for x in list(range(B.length()))]

print('Allocating...',flush=True)
positionsset = AllocateParticlesInBins(B, num_nodes, num_steps, seeds)
print('Allocated', flush=True)

particle_ids = [bin_id(pos, B) for pos in positionsset]
print('Particles with id', particle_ids, 'were generated')

positionsfile = open('positions.data','wb')
pickle.dump(positionsset, positionsfile)
positionsfile.close()
