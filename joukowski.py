#%%
import numpy as np
import matplotlib.pyplot as plt
import spingen


class JoukowskiFlow(spingen.SpinGen):

    def __init__(self, gamma=0):
        super().__init__(gamma)
        self.offset = 0.1 +0.1j
    
    def coodmap(self, z):
        return (z) + self.a**2/(z)

    def quickshow(self):
        fig, ax = plt.subplots(2,2)
        t = np.linspace(0,2*np.pi, 100, dtype=complex)
        dist = np.abs(self.offset - 1)
        t = self.offset + dist*(np.cos(np.real(t)) + 1j*np.sin(np.real(t)))
        time = np.arange(0,3.5,self.dt)
        particles = np.empty((len(time),50),dtype=complex)
        particles[0] = -2.5 + 1j*np.linspace(-3,3,len(particles[0]),dtype=complex)
        for i in range(len(time)-1):
            particles[i+1] = particles[i] + self.dt*self._particle_velocityfield(particles[i])
        for axs in ax:
            for axss in axs:
                axss.plot(np.real(self.coodmap(t)), np.imag(self.coodmap(t)))
                axss.plot(np.real(t), np.imag(t), color="gray")
                axss.set_aspect(1)
        tograph = [0]*4
        for i in range(len(tograph)):
            tograph[i] = particles[int(i*len(particles)/len(tograph))]
            tograph[i] = self.coodmap((tograph[i])*dist+self.offset)
        idl = int(len(particles)/4)
        ax[0,0].scatter(np.real(tograph[0]), np.imag(tograph[0]))
        ax[0,0].scatter(np.real(particles[0]), np.imag(particles[0]), color="gray")
        ax[0,1].scatter(np.real(tograph[1]), np.imag(tograph[1]))
        ax[0,1].scatter(np.real(particles[1*idl]), np.imag(particles[1*idl]), color="gray")
        ax[1,0].scatter(np.real(tograph[2]), np.imag(tograph[2]))
        ax[1,0].scatter(np.real(particles[2*idl]), np.imag(particles[2*idl]), color="gray")
        ax[1,1].scatter(np.real(tograph[3]), np.imag(tograph[3]))
        ax[1,1].scatter(np.real(particles[-1]), np.imag(particles[-1]), color="gray")
        
        plt.show()
jkflow = JoukowskiFlow(0)
jkflow.offset = 0.1+0.1j
jkflow.quickshow()


# %%
