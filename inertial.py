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
        positions[0] = self.xintval[0] + 1j*np.linspace(self.xintval[-1], self.xintval[0], nb_particles)#0 + self.random_initial_positions(nb_particles)
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
                    ycol1 = np.sqrt(np.abs(self.a**2-xcol1**2))*(np.imag(position)/abs(np.imag(position)))
                    ycol2 = np.sqrt(np.abs(self.a**2-xcol2**2))*(np.imag(position)/abs(np.imag(position)))
                    arg = np.argmin(np.array([
                        np.abs(position-(xcol1+1j*ycol1)), np.abs(position-(xcol2+1j*ycol2))]))
                    col = (xcol1+1j*ycol1, xcol2+1j*ycol2)[arg]
                    ### 2.  Law of reflection of the particle's path against the surface of the circle.
                    ###     Calculate the circle's local tangent and use this to update the velocity
                    pretheta = np.arctan2(np.imag(col), np.real(col)) + np.pi/2
                    pretheta = pretheta % np.pi
                    theta = pretheta
                    if theta > np.pi/2:
                        theta -= np.pi
                    phi = np.arctan2(np.imag(velocity), np.real(velocity)) # previous trajectory of the particle
                    newphi = (phi - 2*theta)
                    self.velocity[frame,i] = np.abs(self.velocity[frame,i])*(np.cos(newphi)-1j*np.sin(newphi))
                    # this 0.5 coefficient is arbirtary; change later
                    positions[frame+1, i] = col + 0.5*self.velocity[frame,i]*dt

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
        t = np.linspace(0,2*np.pi, 100)
        coordinates = self.coordinates
        for i in range(4):
            fig, ax = plt.subplots(1)
            ax.set_aspect(1)
            self.plot_stream(ax)
            ax.plot(self.a*np.cos(t), self.a*np.sin(t), color="black")
            T = min(int(len(self.coordinates)/4*i), len(self.coordinates)-1)
            particules = ax.scatter(np.real(self.coordinates[T,::4]), \
                np.imag(self.coordinates[T,::4]), s=20, color="red")
            particules.set_zorder(10)
            ax.set_xlim([self.xintval[0]*1.1,self.xintval[1]*1.1])
            ax.set_ylim([self.xintval[0]*1.1,self.xintval[1]*1.1])
            plt.show()
        fig, ax = plt.subplots(1)
        ax.set_aspect(1)
        self.plot_stream(ax)
        ax.plot(self.a*np.cos(t), self.a*np.sin(t), color="black")
        for i in range(100):
            p =min(int(len(coordinates[0])/25*i), len(coordinates[0])-1)
            trajectoires = ax.plot(np.real(coordinates[:,p]), np.imag(coordinates[:,p]),\
                 linestyle="dashed", color="red",linewidth=8, zorder=10)
        ax.set_xlim([self.xintval[0]*1.1,self.xintval[1]*1.1])
        ax.set_ylim([self.xintval[0]*1.1,self.xintval[1]*1.1])
        plt.show()

if __name__ == "__main__":
    flow = InerMation()
    flow.v0 = 1
    print(flow.streams.shape)
    flow.show_movie(1, 100, 1+0j)