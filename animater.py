import flowgen
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FlowMation(flowgen.FlowGen):

    def __init__(self):
        super().__init__()
        self.velocity = None
    
    def random_initial_positions(self, nb_particles):
        position = np.zeros(nb_particles, dtype=complex)
        position += np.random.uniform(-0.2,0.2,nb_particles) + 1*self.xintval[0]
        position += 1j*np.random.uniform(self.xintval[0], self.xintval[1],nb_particles)
        return position

    def velocityfield(self, positions):
        velocity = 0 + 0j
        r = np.abs(positions)
        velocity += self.v0*(1 + self.a**2/r**2 - 2*(self.a**2)*(np.real(positions)**2)/(r**4))
        velocity += -1j*self.v0/(r**4)*2*self.a*np.real(positions)*np.imag(positions)
        return velocity

    def generate_trajectories(self, nb_frames, nb_particles, dt):
        """ create a matrix of complex-valued coordinates that advance iteratively by time step"""
        positions = np.zeros((nb_frames, nb_particles), dtype=complex) # frame number, particle, (x,1j*y)
        positions[0,:] = self.random_initial_positions(nb_particles)
        self.velocity = np.zeros((nb_frames, nb_particles), dtype=complex)
        
        if self.schema == "FE":
            for frame in range(nb_frames-1):
                #velocity += self.velocityfield(positions[frame])/self.particle_mass
                self.velocity[frame] = self.velocityfield(positions[frame])
                positions[frame+1] = positions[frame] + self.velocity[frame]*dt
            
        
        elif self.schema == "RK4":
            for frame in range(nb_frames-1):
                k1 = self.velocityfield(positions[frame])
                k2 = self.velocityfield(positions[frame] + dt*k1/2)
                k3 = self.velocityfield(positions[frame] + dt*k2/2)
                k4 = self.velocityfield(positions[frame] + dt*k3)
                self.velocity[frame] = (k1+2*k2+2*k3+k4)/6
                positions[frame+1] = positions[frame] + self.velocity[frame]*dt

        return positions

    def show_movie(self, nb_frames=100, nb_particles = 2):
        """
        1. pre-render the trajectories of the particles
        2. plot background (streamlines, cylinder)
        3. build animation
        """
        dt = 0.1
        coordinates = self.generate_trajectories(nb_frames, nb_particles, dt)
        f0, ax = plt.subplots()
        self.plot_stream(ax)
        # plot the cylinder
        t = np.linspace(0, 2*np.pi, 100)
        fig = ax.plot(self.a*np.cos(t), self.a*np.sin(t), color="black")
        
        carte = ax.scatter(np.real(coordinates[0,:]), np.imag(coordinates[0,:]), s=10, color="red") #4.5, "gray"
        carte.set_zorder(10)
        ax.set_xlim([self.xintval[0]*1.1,self.xintval[1]*1.1])
        ax.set_ylim([self.xintval[0]*1.1,self.xintval[1]*1.1])
        ax.set_aspect(1)
        framerate = 1
        
        def updateData(frame):
            stack = np.column_stack(( np.real(coordinates[framerate*frame]),
                np.imag(coordinates[framerate*frame])))
            carte.set_offsets(stack)
            
            return carte

        anime = animation.FuncAnimation(
            f0, updateData, blit=False, frames=coordinates.shape[0], interval=1, repeat=True)
        plt.show()
        plt.close()
    

    def plot_stream(self, ax):          # good
        for equiline in self.equiphi:
            x = np.real(equiline)
            y = np.imag(equiline)
            ax.plot(x,y, color = "gray")
            ax.plot(x, -1*y, color = "gray")
        for stream in self.streams:
            x = np.real(stream)
            y = np.imag(stream)
            if (np.max(np.abs(np.imag(stream))) <= (self.xintval[1] - self.xintval[0])):
                ax.plot(x, y, color="C0")
        ax.set_zorder(0)
        ax.set_aspect(1)

if __name__ == "__main__":
    flow = FlowMation()
    print(flow.streams.shape)
    flow.show_movie(100, 100)