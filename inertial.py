import flowgen
import animater
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class InerMation(animater.FlowMation):
    def __init__(self):
        super().__init__()
        self.tau = 1

    def velocityfield(self, positions, prev_velocity):
        fluidvelo = super().velocityfield(positions)
        inertialvelo = prev_velocity + self.dt*(fluidvelo - prev_velocity)/self.tau
        return inertialvelo
    

    def generate_trajectories(self, nb_frames, nb_particles):
        dt = self.dt
        """ create a matrix of complex-valued coordinates that advance iteratively by time step"""
        positions = np.zeros((nb_frames, nb_particles), dtype=complex) # frame number, particle, (x,1j*y)

        positions[0] = self.random_initial_positions(nb_particles)
        
        
        for frame in range(nb_frames-1):
            #velocity += self.velocityfield(positions[frame])/self.particle_mass
            self.velocity[frame+1] = self.velocityfield(positions[frame], self.velocity[frame])
            positions[frame+1] = positions[frame] + self.velocity[frame]*dt
            

        return positions

    def show_movie(self, nb_frames=100, nb_particles = 2, init_velocity=None):
        self.velocity = np.zeros((nb_frames, nb_particles), dtype=complex)
        if init_velocity is not None:
            self.velocity[0] = init_velocity
        else:
            self.velocity[0] = 0

        super().show_movie(nb_frames, nb_particles)

if __name__ == "__main__":
    flow = InerMation()
    print(flow.streams.shape)
    flow.show_movie(100, 100, 0)
