#Reinjection turned on.

"""
RunAla.py - Runs weighted ensemble on Alanine Dipeptide with a reinjection process
          - Requires that BuildCorseModel.py and AllocateParticles.py have been run
"""
import numpy as np
import pickle
from math import pi as π
from copy import deepcopy

from random import randint

from pathos.multiprocessing import ProcessPool

import sys
sys.path.append("../")


import WeightedEnsemble as WE
import Alanine as Ala
if __name__ == '__main__':
    num_nodes = 40
    n_particles = 120
    n_we_runs = 10**2
    n_we_iters = 10**1
    stepsize = 10**6

    #Reading in bin data, this is generated in AllocateParticles.py
    Bfile = open('BBins.data','rb')
    B0 = pickle.load(Bfile)
    Bfile.close()

    #Reading in particle positions data, this is generated in AllocateParticles.py
    positionsfile = open('positions.data','rb')
    positionsset = pickle.load(positionsfile)
    positionsfile.close()

    #Finding Spawn configuration, looks for the bin closest to the point π/3, π/3
    #and pulls a positions for that bin.
    target_spawn_angles = [-(3*π)/4, (4*π)/5]
    spawn_configuration = Ala.generate_spawn_configuration(target_spawn_angles, positionsset)

    #Building Ensemble
    E0 = Ala.build_ensemble(spawn_configuration, n_particles)
    E0.update_bin_id(B0,Ala.bin_id)
    B0.update_bin_weights(E0)

    #We need target bins for the reinjection process, target location is defined in alanine file
    u, target_binids = Ala.find_target_bins(B0)
    Tfile = open('TMatrix.data','rb')
    T = pickle.load(Tfile)

    #This gives a respawn function that has the target bins and spawn configuration coded into it
    respawnλ = lambda E, B: Ala.respawn(E, B, target_binids, spawn_configuration)

    #building values vectors, u has 1's on the target bins and zeros elsewhere
    print('Building value vectors', flush = True)
    if sum(u)==0:
        print("No Target Bins Found, Refine Coarse Model or Expand Target", flush=True)
    vvals = WE.value_vectors(n_we_iters,T,np.array(u), tol=1.0e-15)
    print('Built', flush = True)
    estimate = []

    #Running WE n_we_runs times
    pool = ProcessPool(nodes = num_nodes)

    print('Running Weighted Ensemble', n_we_runs, 'times with ',n_we_iters,'iterations per run.', flush = True)
    for k in range(n_we_runs):
        E = deepcopy(E0)
        B = deepcopy(B0)
        #weight_flux, num_target_hits, = WE.run_we_reinjection(E, B, vvals, n_we_iters, Ala.mutation, WE.Systematic, Ala.bin_id, respawnλ)
        #weight_flux, num_target_hits, = WE.run_we_reinjection_parallel(E, B, vvals, n_we_iters, Ala.mutation, WE.Systematic, Ala.bin_id, respawnλ, pool)
        #                                        run_we_reinjection_parallel(E, B, vvals, n_we_iters, mutation, resampler, bin_id, reinject, pool)

        weight_flux_total = 0
        num_target_hits_total = 0
        for j in range(n_we_iters):
            v = vvals[:,j] #how does running a non equilibrium sampler affect vvals
            #print('E.bin = ', E.bin)
            #E.resample(B, v, WE.Systematic)
            E.ξ̂  = deepcopy(E.ξ)
            E.ω̂ = deepcopy(E.ω)


            seeds = [randint(1,2**16) for j in range(E.length())]
            stepsizes = [stepsize for j in range(E.length())]
            indices = [j for j in range(E.length())]

            #parallelised mutation step
            output = pool.map(E.mutate_j, indices, [Ala.mutation for j in range(E.length())], stepsizes, seeds)
            E.ξ = [output[j][0] for j in range(len(output))]
            E.ω = np.array([output[j][1] for j in range(len(output))])
            E.update_bin_id(B, Ala.bin_id)

            B.update_bin_weights(E)
            #update time averaged observable
            #if j == n_we_iters-1:
            E, B, num_target_hits, weight_flux = respawnλ(E, B)
            E.update_bin_id(B, Ala.bin_id)
            B.update_bin_weights(E)

            weight_flux_total = weight_flux_total + weight_flux
            num_target_hits_total = num_target_hits_total + num_target_hits

        weight_flux = weight_flux_total
        num_target_hits = num_target_hits_total






        print('Run ',k+1,' of ',n_we_runs,' complete.',' estimate = ',weight_flux, 'num_target_hits = ',num_target_hits,flush = True)
        #I think I need to be passing seeds to the mutation step.
        #print('num_target_hits = ', num_target_hits)
        estimate.append(weight_flux)

    #print('Estimated: ',estimate)
    DMCestimatefile = open('Est_DMC_yes_reinjection.data','wb')
    pickle.dump(estimate, DMCestimatefile)
    DMCestimatefile.close()
