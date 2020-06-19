"""
BuildCorseModel.py - Builds a Transition matrix for the coarse model
                   - Imports Bin data from BBins.data
                   - Runs in parallel
                   - saves transition matrix T to TMatrix.data
                   - requires that AllocateParticles.py has already been run;
                     the bins are build in AllocateParticles.py
"""
import sys
sys.path.append("C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin")
from pathos.multiprocessing import ProcessPool
import numpy as np
import pickle
sys.path.append("../")

if __name__ == '__main__':
    import Alanine as Ala
    from WeightedEnsemble import Bins

    #Number of nodes to run parallel processes on
    num_nodes = 4

    #Readingin bin data, this is generated in AllocateParticles.py
    Bfile = open('BBins.data','rb')
    B = pickle.load(Bfile)
    Bfile.close()

    #Reading in particle position data
    positionsfile = open('positions.data','rb')
    positionsset = pickle.load(positionsfile)
    positionsfile.close()

    #Sampling trajectories from specified positions to build coarse model
    num_samples_per_bin = 10**2
    num_steps = 10**2

    print('Building coarse model...',flush = True)
    n_bins = B.length()

    num_steps = [num_steps for x in range(num_samples_per_bin)]
    seed_data = [x for x in range(num_samples_per_bin)]
    Transitions = []

    particle = Ala.AlanineDipeptideSimulation()
    particle.Bins = B
    particle.temperature = 300
    #particle.stepsize = 1*picoseconds

    T = np.zeros([n_bins,n_bins])
    pool = ProcessPool(nodes = num_nodes)
    for j in range(n_bins):

        #for i in range(num_samples_per_bin):
        particle.positions = positionsset[j]
        Transitions.append(pool.map(particle.sample, num_steps, seed_data))

        #samp=particle.sample(num_steps[j], seed_data[j])
        #Transitions.append(samp)
        #print(samp)


        #print('done', j, 'of', n_bins, 'bins')
        print('Transitions[j] = ', Transitions[j])



    #print(Transitions)
    for j in range(len(Transitions)):
        for i in range(len(Transitions[j])):
            T[Transitions[j][i][0], Transitions[j][i][1]] = T[Transitions[j][i][0], Transitions[j][i][1]] + 1

    for j in range(n_bins):
        if (sum(T[j,:])==0):
            print('No transitions in row', j)
            T[j,j]=1

        T[j,:] = T[j,:] / sum(T[j,:])



    print('Built',flush=True)

    Transitionsfile = open('Transitions.data','wb')
    pickle.dump(Transitions, Transitionsfile)
    Transitionsfile.close()

    Tfile = open('TMatrix.data','wb')
    pickle.dump(T, Tfile)
    Tfile.close()
