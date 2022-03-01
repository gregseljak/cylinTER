#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
class FlowGen():


    def __init__(self):
        self.v0 = 1
        self.res = 10000
        self.a = 1
        self.radlen = int(self.res*np.sqrt(2))
        self.psi = self.psi_from_v(np.linspace(1.0001, 1.99, 20))
        self.phi = np.linspace(0,1,5)
        self.streams = np.empty((len(self.psi), self.radlen), dtype=complex)
        self.equiphi = np.empty((len(self.psi), self.radlen), dtype=complex)
        self.v0 = 1
        self.particle_mass = 1
        self.populate_equilines()
        self.schema = "RK4" # or "FE" or "RE"
        self.schema_dict = {"RK4": "Runge-Kutta 4",
            "FE":"Forward-Euler", "RE":"Reverse-Euler"}
    
    def populate_equilines(self):
        """ Find points along streams and equipotentials for plotting"""
        phi = np.linspace(-20,20,len(self.psi))
        for i in range(len(self.psi)):
            radii = np.linspace(self.a+ 0.001, 15, self.radlen, dtype=np.double)
            y = self.psi[i]/(radii**2-(self.a)**2)*radii**2
            x = np.sqrt(radii**2-y**2)
            self.streams[i,:] = x + 1j*y

            plotpoints = np.empty(len(radii))
            theta = np.arccos(phi[i]/self.v0*radii/(radii**2 + 1))
            plotpoints = radii * np.cos(theta) + 1j* radii * np.sin(theta)
            plotpoints = plotpoints[np.argsort(np.real(plotpoints))] # sort by real part
            self.equiphi[i] = plotpoints


    def psi_from_v(self, velocity):
        """ Recover value of psi given a velocity;
        allows for equal spacing of streamlines to indicate
        velocity gradient """
        y0 = self.a*np.sqrt(self.v0/(velocity-self.v0))
        psi = self.v0 * (1-(self.v0/(velocity-self.v0)))*np.sqrt(self.v0/(velocity-self.v0))*self.a
        for xintcp in y0:
            if (xintcp < self.a):
                continue
        return self.antisymmetrize(psi)
    
    def antisymmetrize(self, array):
        antiarray = -1*np.flip(array)
        return np.append(antiarray, array)

    def plot_stream(self, ax):
        for equiline in self.equiphi:
            x = np.real(equiline)
            y = np.imag(equiline)
            ax.plot(x,y, color = "gray")
            ax.plot(x, -1*y, color = "gray")
        for stream in self.streams:
            x = np.append(-1*np.flip(np.real(stream)), np.real(stream))
            y = np.append(np.flip(np.imag(stream)), np.imag(stream))
            ax.plot(x, y, color="C0")
            x = np.nan_to_num(x, nan=0)
            badsection = np.where(x == 0)[0]
            if (badsection.shape[0] < 2*len(stream)-2):
                b0, bf = badsection[0]-1, badsection[-1]+1
                if ((abs(y[b0]) >= self.a) and (abs(y[bf]) >= self.a)): 
                    ax.plot(np.array([x[b0], x[bf]]), np.array([y[b0], y[bf]]), color="C0", linestyle="-")
        ax.set_zorder(0)
        ax.set_aspect(1)

    def velocityfield(self, instant_positions):
        """takes complex-valued 1d numpy array
            returns d(phi)/dz evaluated at those points"""
        velocity = np.zeros(instant_positions.shape, dtype=complex)
        r = np.abs(instant_positions)
        velocity += self.v0*(1 + self.a**2/r**2 - 2*(self.a**2)*(np.real(instant_positions)**2)/(r**4))
        velocity += -1j*self.v0/(r**4)*2*self.a*np.real(instant_positions)*np.imag(instant_positions)
        return velocity

    def generate_trajectories(self, nb_frames, nb_particles, dt, initial_positions=None):
        """ create a matrix of complex-valued coordinates that advance iteratively by time step"""
        positions = np.zeros((nb_frames, nb_particles), dtype=complex) # frame number, particle, (x,1j*y)
        if initial_positions:
            positions[0,:] = initial_positions
        else:
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

    def random_initial_positions(self, nb_particles):
        position = np.zeros(nb_particles, dtype=complex)
        position += np.random.uniform(-0.2,0.2,nb_particles) - 1*self.xspan
        position += 1j*np.random.uniform(-3,3,nb_particles)
        return position

    def schema_evaluation(self):
        pass

    def show_movie(self, nb_frames=100, nb_particles = 2):
        """
        1. pre-render the trajectories of the particles
        2. plot background (streamlines, cylinder)
        3. build animation
        """
        self.xspan = 5
        dt = 0.1
        coordinates = self.generate_trajectories(nb_frames, nb_particles, dt)
        f0, ax = plt.subplots()
        self.plot_stream(ax)
        # plot the cylinder
        t = np.linspace(0, 2*np.pi, 100)
        fig = ax.plot(self.a*np.cos(t), self.a*np.sin(t), color="black")
        
        carte = ax.scatter(np.real(coordinates[0,:]), np.imag(coordinates[0,:]), s= 10, color="red") #4.5, "gray"
        carte.set_zorder(10)
        ax.set_xlim([-1*self.xspan*1.1,self.xspan*1.1])
        ax.set_ylim([-1*self.xspan*1.1,self.xspan*1.1])
        ax.set_aspect(1)
        framerate = 1
        def updateData(frame):
            stack = np.column_stack(( np.real(coordinates[framerate*frame]),
                np.imag(coordinates[framerate*frame])))
            carte.set_offsets(stack)
            
            return carte

        anime = animation.FuncAnimation(
            f0, updateData, blit=False, frames=coordinates.shape[0], interval=1, repeat=True)
        # f0.tight_layout()
        plt.show()
        plt.close()

def main():
    flow = FlowGen()
    flow.show_movie(1000,50)
    return flow
    #flow.show_movie(100, nb_particles=50)

if __name__ == "__main__":
    flow = main()
