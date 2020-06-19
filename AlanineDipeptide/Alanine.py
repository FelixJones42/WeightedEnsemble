"""
Alanine.py - Python module.
           - Includes the AlanineDipeptideSimulation class
           - Includes functions for generating the coarse model
           - Includes functions for running the reinjection WE process

"""
import numpy as np
from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d
from pathos.multiprocessing import ProcessPool
import random
from random import randint
from math import pi as π
import pickle

from simtk import openmm, unit
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from openmmtools import testsystems
from sys import stdout
import mdtraj

import sys
sys.path.append("../")

import WeightedEnsemble as WE
from WeightedEnsemble import Bins

"""
Alanine Molecule Class
"""

class AlanineDipeptideSimulation():
    def __init__(self):
        self.positions = testsystems.AlanineDipeptideVacuum().positions.value_in_unit(nanometer)
        self.topology = testsystems.AlanineDipeptideVacuum().topology
        self.system = testsystems.AlanineDipeptideVacuum().system
        self.Bins = [] #only used for sampling, storing it here saves memory when calling pmap
        self.temperature = 300 #Kelvin
        self.stepsize = 0.002*picoseconds

    def minimize_energy(self):
        integrator = LangevinIntegrator(self.temperature*kelvin, 1/picosecond, self.stepsize)
        simulation = Simulation(self.topology,self.system,integrator)
        simulation.context.setPositions(self.positions)
        simulation.minimizeEnergy()
        state = simulation.context.getState(getPositions=True)
        self.positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)

    def get_energy(self):
        integrator = LangevinIntegrator(self.temperature*kelvin, 1/picosecond, self.stepsize)
        simulation = Simulation(self.topology,self.system,integrator)
        simulation.context.setPositions(self.positions)
        state = simulation.context.getState(getEnergy=True)
        return state.getPotentialEnergy()

    def Step(self, num_steps, seed=0):
        integrator = LangevinIntegrator(self.temperature*kelvin, 1/picosecond, self.stepsize)
        #integrator.setRandomNumberSeed(seed)
        simulation = Simulation(self.topology,self.system,integrator)
        simulation.context.setPositions(self.positions)
        simulation.step(num_steps)
        state = simulation.context.getState(getPositions=True)
        self.positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)

    def getAngles(self):
        traj = mdtraj.Trajectory(self.positions, mdtraj.Topology.from_openmm(self.topology))
        psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
        angles = mdtraj.compute_dihedrals(traj, [phi_indices, psi_indices])
        #print('compute_dihedrals returns', angles[0], ' with positions ')
        return angles[0]

    def bin_id(self, B):
        angles = self.getAngles()
        j=0
        bin_id = -1
        while j < B.length():
            if angles[0] < B.Ω[j][1][0]:
                if angles[1] < B.Ω[j][1][1]:
                    bin_id = j
                    j=B.length()+1
            j=j+1;

        #if bin_id == -1:
            #print('angles = ', angles)

        return bin_id

    def bin_id_voronoi(self, B):
        tree = KDTree(B.Ω)
        d, id  = tree.query(self.positions)
        return id

    def AddTorsionForce(self, phi, psi, magnitude):
        torsion_force = openmm.CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0)")
        torsion_force.addPerTorsionParameter("k")
        torsion_force.addPerTorsionParameter("theta0")
        torsion_force.addGlobalParameter("pi",3.141592653589793)
        k = magnitude * kilojoule
        torsion_force.addTorsion(6,8,14,16, [k, psi])
        torsion_force.addTorsion(4, 6, 8, 14, [k, phi])
        self.system.addForce(torsion_force)

    def RemoveTorsionForce(self, phi, psi):
        self.system = testsystems.AlanineDipeptideVacuum().system

    def move_particle_to_bin_minenergy(self, phi, psi, magnitude):
        #This isn't seeded, but perhaps it should be
        self.AddTorsionForce(phi, psi, magnitude)
        self.minimize_energy()
        self.RemoveTorsionForce(phi, psi)
        return self.positions

    def move_particle_to_bin_minenergy_with_stepping(self, phi, psi, magnitude, seed, B, target_id):
        #This isn't seeded, but perhaps it should be
        self.AddTorsionForce(phi, psi, magnitude)
        self.minimize_energy()

        id = -2
        count = 0
        while id != target_id and count < 10**2:
            self.Step(num_steps, seed)
            id = self.bin_id(B)
            if id == -1:
                id = target_id

            count = count +1

        self.RemoveTorsionForce(phi, psi)
        return self.positions


    def move_particle_to_bin_stepping_old(self, phi, psi, num_steps, seed, B, target_id):
        #Obsolete, moves particle to bin by adding torsion force and stepping
        #We use the minimize energy approach now instead
        self.AddTorsionForce(phi, psi)
        id = -2
        count = 0
        while id != target_id and count < 10**2:
            self.Step(num_steps, seed)
            id = self.bin_id(B)
            if id == -1:
                id = target_id

            count = count +1
            #if count == 10**2:
            #    print('Bin ' , target_id, ' failed to allocate', flush=True)

        self.RemoveTorsionForce(phi, psi)
        #print('Bin ',target_id,' took ', count, 'iterations to get a position', flush = True)
        return self.positions

    def sample(self, num_steps, seed):
        start_bin = self.bin_id(self.Bins)
        self.Step(num_steps, seed)
        end_bin = self.bin_id(self.Bins)
        return [start_bin, end_bin]

    def sample_voronoi(self, num_steps, seed):
        start_bin = self.bin_id_voronoi(self.Bins)
        self.Step(num_steps, seed)
        end_bin = self.bin_id_voronoi(self.Bins)
        return [start_bin, end_bin]
"""
Functions for generating the coarse model

"""

def BinMidpoints(B, bin_no):
    ϕ = (B.Ω[bin_no][0][1]+B.Ω[bin_no][0][0])/2
    ψ = (B.Ω[bin_no][1][1]+B.Ω[bin_no][1][0])/2
    return [ϕ, ψ]

def Build_UnifBins(nx_bins, ny_bins):

    xx = np.linspace(-π, π, nx_bins+1)
    yy = np.linspace(-π, π, ny_bins+1)
    B = Bins()

    for j in range(nx_bins):
        for k in range(ny_bins):
            Ω = np.array([[xx[j],yy[k]],[xx[j+1],yy[k+1]]])#Identifying rectangular bins by a pair of opposite corners.
            B.push(Ω,0)

    B.dim = [nx_bins,ny_bins]

    return B

def Build_Voronoi_Bins(positions, energy_threshold):
    """
    Requires a selection of particle positions, such as would be generated by Build_UnifBins
    and AllocateParticlesInBins (positionsset).

    Particles above the energy threshold are removed.

    A bin structure element here is a particle, the work is done in the bin_id_voronoi method
    of the AlanineDipeptideSimulation class. Here bins are identified based on which one
    of the particles in B.Ω is closest.
    """
    B = Bins()

    #for j in range(np.shape(positions)[0]):

def AllocateParticlesInBins(B, num_nodes, step_data):
    #seeds not being used at the moment
    magnitude = 1000.0

    particle_data = []
    phi_data = []
    psi_data = []
    target_ids = []
    B_copies = [B]
    magnitude_copies=[]

    for j in range(B.length()):
        angles = BinMidpoints(B, j)
        phi_data.append(angles[0])
        psi_data.append(angles[1])
        target_ids.append(j)
        B_copies.append(B)
        magnitude_copies.append(magnitude)

    pool = ProcessPool(nodes = num_nodes)
    particle = AlanineDipeptideSimulation()

    #positionsset = pool.map(particle.move_particle_to_bin, phi_data, psi_data, step_data, seeds, B_copies, target_ids)
    positionsset = pool.map(particle.move_particle_to_bin_minenergy, phi_data, psi_data, magnitude_copies)
    #positions = particle.move_particle_to_bin(1,1, step_data[0], seeds[0])

    return positionsset

def build_coarse_model(positionsset, B, num_samples_per_bin, num_nodes, num_steps):
    n_bins = B.length()
    pool = ProcessPool(nodes = num_nodes)

    num_steps = [num_steps for x in range(num_samples_per_bin)]
    seed_data = [x for x in range(num_samples_per_bin)]
    Transitions = []

    particle = AlanineDipeptideSimulation()
    particle.Bins = B
    particle.temperature = 1000
    #particle.stepsize = 1*picoseconds

    T = np.zeros((n_bins,n_bins))

    for j in range(n_bins):
        particle.positions = positionsset[j]
        Transitions.append(pool.map(particle.sample, num_steps, seed_data))

        for j in range(len(Transitions)):
            T[Transitions[j][0], Transitions[j][1]] = T[Transitions[j][0], Transitions[j][1]] + 1

    for j in range(n_bins):
        if (sum(T[j,:])==0):
            #print('No transitions in row', j, flush = True)
            T[j,j]=1

        T[j,:] = T[j,:] / sum(T[j,:])

    return T

def build_coarse_model_voronoi(B, num_samples_per_bin, num_nodes, num_steps):
    n_bins = B.length()
    pool = ProcessPool(nodes = num_nodes)

    num_steps = [num_steps for x in range(num_samples_per_bin)]
    seed_data = [x for x in range(num_samples_per_bin)]
    Transitions = []

    particle = AlanineDipeptideSimulation()
    particle.Bins = B
    particle.temperature = 1000

    T = np.zeros((n_bins,n_bins))

    for j in range(n_bins):
        particle.positions = B.Ω[j]
        Transitions.append(pool.map(particle.sample_voronoi, num_steps, seed_data))

        for j in range(len(Transitions)):
            T[Transitions[j][0], Transitions[j][1]] = T[Transitions[j][0], Transitions[j][1]] + 1

    for j in range(n_bins):
        if (sum(T[j,:])==0):
            #print('No transitions in row', j, flush = True)
            T[j,j]=1

        T[j,:] = T[j,:] / sum(T[j,:])

    return T

def mutation(pos, stepsize, seed):
    sim = AlanineDipeptideSimulation()
    sim.positions = pos
    sim.Step(stepsize, seed)
    return sim.positions


"""
Functions for running the reinjection process

"""

def In_target(ϕ, ψ):
    target_centre = [(54*π)/180,-π/4]
    tolerance = π/9 #copperman suggestion 20 degrees (π/9) but my coarse model isn't fine enough right now
    if target_centre[0]-tolerance < ϕ and ϕ < target_centre[0]+tolerance:
        if target_centre[1]-tolerance < ψ and ψ < target_centre[1]+tolerance:
            return 1
        else:
            return 0
    else:
        return 0

def find_target_bins(B):
    #u contains 1 in the bins with midpoint inside the target area
    midpoints = []
    u=[]
    target_binids = []
    for j in range(B.length()):
        midpoints.append([(B.Ω[j][1][0]+B.Ω[j][0][0]) / 2,(B.Ω[j][1][1]+B.Ω[j][0][1]) / 2])
        u.append(In_target(midpoints[j][0],midpoints[j][1]))
        if u[j] == 1:
            target_binids.append(j)
    #it is possible to have a target structure with no midpoints in the bins.
    print(sum(u),' target bins found')
    if sum(u)==0:
        print("No Target Bins Found, Refine Coarse Model or Expand Target")
    return u, target_binids

def generate_spawn_configuration(target_spawn_angles, positionsset):

    angle_distance = 100
    for j in range(len(positionsset)):
        sim = AlanineDipeptideSimulation()
        sim.positions = positionsset[j]
        angles = sim.getAngles()
        if np.linalg.norm(angles - target_spawn_angles) < angle_distance:
            angle_distance = np.linalg.norm(angles - target_spawn_angles)
            spawn_configuration = positionsset[j]
    return spawn_configuration

def build_ensemble(spawn_configuration, n_particles):
    E0 = WE.Ensemble()
    E0.ξ = [spawn_configuration] * n_particles
    E0.ξ̂ = [spawn_configuration] * n_particles

    E0.ω = np.ones(n_particles) / n_particles
    E0.ω̂ = np.ones(n_particles) / n_particles
    E0.bin = np.ones(n_particles).astype(int)
    return E0

def bin_id(position, B):
    sim = AlanineDipeptideSimulation()
    sim.positions = position
    return sim.bin_id(B)

def bin_id_voronoi(position, B):
    sim = AlanineDipeptideSimulation()
    sim.positions = position
    return sim.bin_id_voronoi(B)

def respawn(E, B, target_binids, spawn_configuration):
    """
    Alanine reinjection process - target_binids are bins where resampling occurs
                                - spawn configuration is the position particles
                                  are respawned with
                                - As well as ensemble and bin structures, this
                                  also returns the number of times the target
                                  was hit and the sum of the weight moving
                                  through the reinjection.
    """
    weight_flux = 0
    num_target_hits = 0
    for j in range(E.length()):
        for k in range(len(target_binids)):
            if E.bin[j] == target_binids[k]:
                weight_flux = weight_flux + E.ω[j]
                E.ξ[j] = spawn_configuration
                num_target_hits = num_target_hits + 1

    return E, B, num_target_hits, weight_flux

def respawn_voronoi(E, B, target_binids, spawn_configuration):
    """
    Alanine reinjection process - target_binids are bins where resampling occurs
                                - spawn configuration is the position particles
                                  are respawned with
                                - As well as ensemble and bin structures, this
                                  also returns the number of times the target
                                  was hit and the sum of the weight moving
                                  through the reinjection.
    """
    weight_flux = 0
    num_target_hits = 0
    for j in range(E.length()):
        for k in range(len(target_binids)):
            if E.bin[j] == target_binids[k]:
                weight_flux = weight_flux + E.ω[j]
                E.ξ[j] = spawn_configuration
                num_target_hits = num_target_hits + 1

    E.update_bin_id(B, bin_id_voronoi)
    B.update_bin_weights(E)
    return E, B, num_target_hits, weight_flux
