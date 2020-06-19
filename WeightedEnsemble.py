"""
WeightedEnsemble.py -   Weighted Ensemble Python module.
                        Includes both the Ensemble and Bins classes

"""
import numpy as np
from pathos.multiprocessing import ProcessPool
from copy import deepcopy
from random import randint


#Builds Ensemble and Bin classes
class Ensemble:
    def __init__(self):
        self.ξ = [] #Particle location before mutation but after resampling
        self.ξ̂ = [] #Particle location after resampling but before mutation
        self.ω = np.array([]) # ξ weight
        self.ω̂ = np.array([]) # ξ̂ weight
        self.bin = [] #bin identifier for particle ξ
        self.mutator = []


    def push(self,ξp,ωp,binp): #I don't think numpy arrays are really the right structure to be pushing?
        self.ξ = np.append(self.ξ, ξp)
        self.ξ̂ = np.append(self.ξ̂, ξp)
        self.ω = np.append(self.ω,ωp)
        self.ω̂ = np.append(self.ω̂,ωp)
        self.bin = np.append(self.bin,binp)

    def length(self):
        return len(self.ξ)

    def update_bin_id(self,B,bin_id): #1D only function
        n_particles = self.length()

        for j in range(n_particles):
            self.bin[j] = bin_id(self.ξ[j],B)

    def mutate(self,mutation):
        #mutations ξ̂ and gives it to ξ
        for j in range(self.length()):
            self.ξ[j] = mutation(self.ξ̂[j],randint(1,2**16))
            self.ω[j] = self.ω̂[j]

    def mutate_j(self, j, mutation, stepsize, seed):
        #used in mutate_parallel
        return mutation(self.ξ̂[j], stepsize, seed) , self.ω̂[j]

    def resample(self, B, v, resampler):
        n_particles = self.length()
        n_bins = B.length()


        #Calulate target number of offspring per bin
        #print('B.ν = ',B.ν)
        resampling_ids = [index for index, value in enumerate(B.ν) if value > 0] #positive bin weight means that bin
        #print('resampling_ids =',resampling_ids)
        #must have a particle in it.
        R = len(resampling_ids) #count number of bins which must have offspring
        #print('R =',R)
        target = np.zeros(n_bins) #are there memory problems with doing this?
        target[resampling_ids] = 1
        #print('target1 =',target)
        if n_particles > R:
            #print('B.ν = ', B.ν)
            #print('sum(v) = ',np.sum(v))
            resampling_weights = np.multiply(B.ν,v) / np.dot(B.ν, v)
            print('sum resampling_weights =',sum(resampling_weights))
            #print('int(n_particles - R) = ',int(n_particles - R))
            target = target + resampler(n_particles - R, resampling_weights )

        #print('target2 =',target)
        #print('resampling_weights =',resampling_weights)

        #if np.sum(target) != n_particles:
        #    print('wrong number of targets')
        #compute number of offspring of each particle bin by bin
        offspring = np.empty(n_particles).astype(int)
        for j in range(n_bins):
            particle_ids = [index for index, value in enumerate(self.bin) if value == j] # id's of particle in bin j
            if len(particle_ids)!=0:
                offspring[particle_ids] = resampler(int(target[j]), self.ω[particle_ids] / B.ν[j])

        #print('target = ',target)
        #print('offspring = ',offspring)
        #resample the particles
        n_spawned = 0
        #for each walker, for each offspring
        #val1 = self.ξ
        #print('target =', target)
        #print('offspring =', offspring)

        #print('particlesum1 = ',np.sum(self.ω̂))
        for j in range(n_particles):
            bin_id = self.bin[j]
            for k in range(offspring[j]):
                #print('triggered')
                #if  k>1:
                    #print('k= ',k)
                    #print('n_spawned = ', n_spawned)
                    #print('self.ξ[j] = ',self.ξ[j])
                    #print('self.ξ̂[k + n_spawned] = ',self.ξ̂[k + n_spawned])

                self.ξ̂[k + n_spawned] = deepcopy(self.ξ[j]) #there is a bloody hat there that is impossible to see
                self.ω̂[k + n_spawned] = B.ν[bin_id] / target[bin_id] #this is the line

                #if  k>1:
                    #print('AFTER ALLOCATION')
                    #print('k= ',k)
                    #print('n_spawned = ', n_spawned)
                    #print('self.ξ[j] = ',self.ξ[j])
                    #print('self.ξ̂[k + n_spawned] = ',self.ξ̂[k + n_spawned])

            n_spawned += offspring[j]
        #print('particlesum2 = ',np.sum(self.ω̂))
        #for j in range(n_particles):
        #    self.ξ[j] = self.ξ̂[j] #NOTE THE HAT
        #    self.ω[j] = self.ω̂[j]#NOTE THE HAT


class Bins:
    def __init__(self):
        self.Ω = [] #Bin structure
        self.ν = [] #Bin weights
        self.dim = []

    def push(self,Ωp,νp):
        self.Ω.append(Ωp)
        self.ν.append(νp)

    def length(self):
        return len(self.Ω)

    def update_bin_weights(self,E):
        #print('update bin weights function')
        n_walker = E.length()
        n_bins = self.length()
        for j in range(n_bins):
            particle_ids = [index for index, value in enumerate(E.bin) if value == j]
            #print('particle_ids = ', particle_ids)
            self.ν[j] = sum(E.ω[particle_ids])
            #self.n[j] = len(particle_ids), we don't use this anywhere
        #print('B.ν = ',self.ν)
        #print('E.bin  = ',E.bin)
def value_vectors(n_we_iters,T,u, tol=1.0e-15):
    """ Compute the value vectors for a WE run

    Given the transition matrix, the coarse objective function, and the number of
    iterations, this computes the associated value vectors needed for the WE run.

    n_we_iters - Number of WE iterations in a single run
    T - Coarse state transition matrix
    u - Coarse state quantity of interest vector
    """

    n_bins = np.size(u)
    vvals = np.zeros([n_bins,n_we_iters])

    Tu = deepcopy(u)
    v1 = np.zeros(n_bins)
    v2 = np.zeros(n_bins)

    for j in range(n_we_iters-1,-1,-1):
        v1 = deepcopy(Tu)
        Tu = T.dot(Tu)
        v2 = deepcopy(Tu)
        v1 = v1 ** 2
        v1 = T.dot(v1)
        v2 = v2 ** 2

        #if np.min(v1-v2) < -tol:
            #print("Min v^2 = ", np.min(v1-v2))

        vvals[:,j] = np.sqrt(np.maximum(v1 - v2,np.zeros(np.size(v1))))

    return vvals


def Systematic(n, ω):

    U = np.arange(n)/n + np.random.rand()/n
    Nvals = np.bincount(np.searchsorted(np.cumsum(ω)/sum(ω), U),minlength= np.size(ω))

    return Nvals


def run_we(E, B, vvals, n_we_iters, mutation, resampler, bin_id, reinject, pool):
    """ Run the WE algorithm

    Runs the WE algorithm for the specified number of iterations.
    This assumes that the ensemble and bin strucutres, E and B, have already been
    properly initialized.

    Runs the mutation step in parallel.

    User must specify a mutation routine, a resampling routine, a bin id scheme
    and a reinjection scheme in addition to the value vectors and the number of
    iterations
    """

    weight_flux_total = 0
    num_target_hits_total = 0
    for j in range(n_we_iters):
        v = vvals[:,j] #how does running a non equilibrium sampler affect vvals
        #print('E.bin = ', E.bin)
        E.resample(B, v, resampler)

        seeds = [randint(1,2**16) for j in range(E.length())]

        pool.map(E.mutate_j, [j for j in range(E.length())], [mutation for j in range(E.length())])

        E.update_bin_id(B, bin_id)

        B.update_bin_weights(E)
        #update time averaged observable
        E, B, num_target_hits, weight_flux = reinject(E, B)
        E.update_bin_id(B, bin_id)
        B.update_bin_weights(E)

        weight_flux_total = weight_flux_total + weight_flux
        num_target_hits_total = num_target_hits_total + num_target_hits

    return weight_flux_total, num_target_hits_total
