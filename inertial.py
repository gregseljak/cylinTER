import flowgen
import animater
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class InerMation(animater.FlowMation):
    def __init__(self):
        super().__init__()
        self.tau = 2

    def velocityfield(self, positions, prev_velocity):
        fluidvelo = super().velocityfield(positions)
        inertialvelo = prev_velocity + self.dt*(fluidvelo - prev_velocity)/self.tau
        return inertialvelo
    

    def generate_trajectories(self, nb_frames, nb_particles):
        
        """ create a matrix of complex-valued coordinates that advance iteratively by time step"""
        positions = np.zeros((nb_frames, nb_particles), dtype=complex) # frame number, particle, (x,1j*y)
        dt = self.dt
        positions[0] = self.random_initial_positions(nb_particles)
        if np.isnan(self.velocity[0,0]):
            self.velocity[0] = super().velocityfield(positions[0])
            print("assumed fluid initial velocity")
        
        for frame in range(nb_frames-1):

            positions[frame+1] = positions[frame] + self.velocity[frame]*dt
            # collision check:
            for i in range(len(positions[frame+1])):
                if (np.abs(positions[frame+1, i]) < self.a): #verboten

                    ### 1. Find values of (x,y) where the collision may occur;
                    ###     interesection of circle and line. Here we assume the
                    ###     min(x1,x2) is correct and the other is extraneous as
                    ###     the collision occurs on the left side of the circle
                    velocity = self.velocity[frame, i]
                    position = positions[frame, i]
                    slope = np.imag(velocity)/np.real(velocity)
                    intcp = np.imag(positions[frame, i]) - np.real(positions[frame, i])*slope
                    a = slope**2+1
                    b = 2*intcp*slope
                    c = intcp**2-1*self.a**2
                    xcol1 = 1/(2*a)*(-1*b+np.sqrt(b**2-4*a*c))
                    xcol2 = 1/(2*a)*(-1*b-np.sqrt(b**2-4*a*c))
                    xcol = min(xcol1, xcol2)
                    ycol = np.sqrt(np.abs(self.a**2-xcol**2))*(np.imag(position)/abs(np.imag(position)))
                    ### 2.  Law of reflection of the particle's path against the surface of the circle.
                    ###     Calculate the circle's local tangent and use this to update the velocity
                    theta = np.arctan(ycol/xcol) - np.pi/2 # tangent of the circle
                    phi = np.arctan(slope)                 # previous trajectory of the particle
                    newphi = 2*theta-phi
                    self.velocity[frame,i] = np.abs(self.velocity[frame,i])*(np.cos(newphi)+1j*np.sin(newphi))
                    # this 0.5 coefficient is arbirtary; change later
                    positions[frame+1, i] = xcol + 1j*ycol + 0.5*self.velocity[frame,i]*dt

            self.velocity[frame+1] = self.velocityfield(positions[frame], self.velocity[frame])
        return positions

    def show_movie(self, dt=0.1, nb_particles = 2, init_velocity=None):
        nbframe = 200
        T = self.xintval[1] - self.xintval[0]
        if dt > T/nbframe:
            print(f"warning; dt = {dt} too large for nbframe {nbframe}")
            print(f"defaulting to {T/nbframe}")
            self.dt = T/nbframe
        else:
            self.dt = dt
        skip = int(T/self.v0/200/self.dt)
        
        self.velocity = np.zeros((nbframe*skip, nb_particles), dtype=complex)
        if init_velocity is not None:
            self.velocity[0] = init_velocity
        else:
            self.velocity[0] = np.nan # to be resolved by generate_trajectories

        super().show_movie(self.dt, nb_particles)

if __name__ == "__main__":
    flow = InerMation()
    flow.v0 = 0.5
    print(flow.streams.shape)
    flow.show_movie(0.001, 200,6)