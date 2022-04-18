#%%
import logging
import matplotlib.pyplot as plt
import numpy as np

#%%
def binsearch(function, domain):
    ydomain = domain
    survey = function(ydomain)
    bigroots = np.where(survey[:-1]*survey[1:] <= 0)[0]
    if len(bigroots) == 0:
        #print(" No roots seen in survey ")
        return np.array([])
    roots = np.empty(len(bigroots))
    for i in range(len(roots)):
        ymin = ydomain[bigroots[i]]
        ymax = ydomain[bigroots[i]+1]
        yguess = (ymin+ymax)/2
        
        cnt,threshold = 0, 1000
        while (abs(ymax-ymin) > 0.001):
            yguess = (ymin+ymax)/2
            if function(yguess)*function(ymin) > 0:
                ymin = yguess
            if function(yguess)*function(ymax) > 0:
                ymax = yguess
            cnt += 1
            if cnt >= threshold:
                print(f"threshold {threshold} reached")
                break
        roots[i] = yguess
    return roots

def roots_to_parametrization(ragged_array):
    rgdarr = ragged_array
    nb_psipts = 0
    for j in range(len(rgdarr)):
        nb_psipts += len(rgdarr[j])
    stream = np.zeros(nb_psipts, dtype=complex)
    xidx = 0
    while len(rgdarr[xidx]) == 0:
        xidx += 1
        if xidx == len(rgdarr)-1:
            return np.array([], dtype=complex)
    arg = np.argmin(np.imag(rgdarr[xidx]))
    stream[0] = (rgdarr[xidx])[arg]
    rgdarr[xidx] = np.delete(rgdarr[xidx], arg)
    xidx += 1
    for j in range(nb_psipts-2):
        #print(f"x = {xidx} j = {j} nbp = {nb_psipts-2}")
        if len(rgdarr[xidx]) == 0:
            xidx = 0
            while len(rgdarr[xidx]) == 0:
                xidx += 1
        arg = np.argmin(np.abs(np.imag(stream[j]) - np.imag(rgdarr[xidx])))
        stream[j+1] = (rgdarr[xidx])[arg]
        rgdarr[xidx] = np.delete(rgdarr[xidx], arg)
        lenleft = 0
        lenright = 0
        if xidx > 0:
            lenleft = len(rgdarr[xidx-1])
        if xidx < len(rgdarr)-1:
            lenright = len(rgdarr[xidx+1])
        lenmiddle = len(rgdarr[xidx])
        dists = np.zeros(3) + 1000
        if lenleft > 0:
            dists[0] = np.min(np.abs(stream[j] - rgdarr[xidx-1]))
        if lenright > 0:
            dists[2] = np.min(np.abs(stream[j] - rgdarr[xidx+1]))
        if lenmiddle > 0:
            dists[1] = np.min(np.abs(stream[j] - rgdarr[xidx]))
        xidx += (np.argmin(dists)-1)
    return stream[:-1]

class SpinGen():


    def __init__(self, gamma=0):
        self.equiphi = None
        self.streams=None
        self.v0 = 1
        self.gamma = gamma
        self.xintval = np.array([-6,6])
        self.res = 1000
        self.a = 1
        self.psi = np.linspace(0.2,0.9,10)#np.linspace(0.0001,0.99,10)
        if self.gamma == 0:
            self.psi = self.antisymmetrize(self.psi)
        self.phi = np.linspace(-10,10, 9)
        self.dt = 0.001
        self.streams = np.empty((len(self.psi), self.res), dtype=complex)
        self.equiphi = np.empty((len(self.psi), self.res), dtype=complex)
        #self.populate_equilines()
        self.schema = "RK4" # or "FE" or "RE"
        self.schema_dict = {"RK4": "Runge-Kutta 4",
            "FE":"Forward-Euler", "RE":"Reverse-Euler"}
    
    def make_array(self, args):
        data = []
        for arg in args:
            if arg is None:
                data.append(None)
            elif (not isinstance(arg, np.ndarray)):
                data.append(np.ndarray(arg), dtype=complex)
            else:
                data.append(arg)
        return data

    def populate_equilines(self):
        """ Find points along streams and equipotentials for plotting"""
        streams = [0]*len(self.psi)
        equiphis = [0]*len(self.phi)
        x = np.linspace(5*self.xintval[0], 5*self.xintval[1], 1000)
        psisol = self.y_from_psix(self.psi, x)
        phisol = self.y_from_phix(self.phi, x)
        #print(f"psisol.shape: {psisol.shape}")
        print(f"phisol.shape: {phisol.shape}")
        #print (f"psisol first vals: {psisol[0]}")
        #print (f"psisol first vals: {phisol}")
        for i in range(len(psisol)):
            streams[i] = roots_to_parametrization(psisol[i])
        for i in range(len(phisol)):
            equiphis[i] = roots_to_parametrization(phisol[i])
        self.streams = streams
        self.equiphi = equiphis
        

    def _equipsi_fn(self,psi,x):

        def subordinate_equipsi(y):
            return psi - (self.v0*(y-self.a**2*y/(x**2+y**2) + self.gamma/(2*np.pi)*np.log(np.abs(x+1j*y))))   
        return subordinate_equipsi

    def y_from_psix(self, inpsi = None, xvals_in = None, innerstream=False):
        # Given a populated np.array of psi and x, find y
        npsi = self.psi
        xvals = np.linspace(self.xintval[0], self.xintval[1], 1000)
        uservals = self.make_array((inpsi, xvals_in))
        if uservals[0] is not None:
            npsi = uservals[0]
        if uservals[1] is not None:
            xvals = uservals[1]
            #quit()
            

        solution = np.empty((len(npsi), len(xvals)), dtype=np.ndarray)
        for i in range(len(npsi)):
            psi = npsi[i]
            for j in range(len(xvals)):
                x = xvals[j]
                equipsi = self._equipsi_fn(psi,x)
                roots = binsearch(equipsi, 1.5*np.linspace(-8, 8, 1000))
                if not innerstream:
                    roots = np.delete(roots, np.where(np.abs(x+1j*roots) < self.a)[0])
                #if len(roots) == 0:
                    #print(f" y_from_psix problem at psi = {psi}, x = {x}")
                solution[i,j] = x + 1j*roots
        return solution
    
    def y_from_phix(self, inphi = None, xvals_in = None):
        nphi = self.phi
        xvals = np.linspace(self.xintval[0], self.xintval[1], 1000)
        uservals = self.make_array((inphi, xvals_in))
        if uservals[0] is not None:
            nphi = uservals[0]
        if uservals[1] is not None:
            xvals = uservals[1]

        solution = np.empty((len(nphi), len(xvals)), dtype=np.ndarray)
        for i in range(len(nphi)):
            phi = nphi[i]
            for j in range(len(xvals)):
                x = xvals[j]
                equipsi = self._equiphi_fn(phi,x)
                roots = binsearch(equipsi, 1.5*np.linspace(-8, 8, 1000))
                roots = np.delete(roots, np.where(np.abs(x+1j*roots) < self.a)[0])
                solution[i,j] = x + 1j*roots
        return solution

    def _equiphi_fn(self,phi,x):

        def subordinate_equiphi(y):
            return phi - (self.v0*(x+self.a**2*x/(x**2+y**2) - self.gamma/(2*np.pi)*np.arctan2(y, x)))   
        return subordinate_equiphi



    def psi_from_v(self, velocity): #checked: good
        """ Recover value of psi given a velocity;
        allows for equal spacing of streamlines to indicate
        velocity gradient """
        y0 = self.a*np.sqrt(self.v0/(velocity-self.v0))
        psi = self.v0 * (1-(self.v0/(velocity-self.v0)))*np.sqrt(self.v0/(velocity-self.v0))*self.a
        for xintcp in y0:
            if (xintcp < self.a):
                continue
        return self.antisymmetrize(psi)

################## Should be good from here #################################      
    def antisymmetrize(self, array):    #checked: good
        antiarray = -1*np.flip(array)
        return np.append(antiarray, array)

    def _particle_velocityfield(self, u):
        """takes complex-valued singleton numpy array
            returns d(phi)/dz evaluated at that"""
        velocity = 0 + 0j
        r = np.abs(u)
        velocity += self.v0*(1 + self.a**2/r**2 - 2*(self.a**2)*(np.real(u)**2)/(r**4))
        velocity += self.gamma/(2*np.pi)*(np.imag(u)/(r**2))
        velocity += -1j*self.v0/(r**4)*2*self.a*np.real(u)*np.imag(u)
        velocity += -1j*(self.gamma/2/np.pi)*np.real(u)/(r**2)
        return velocity

    def xintval_integration(self, mindx = 0.001):
        """ create a matrix of complex-valued coordinates that advance iteratively by time step"""
        dt = self.dt
        positions = None
        y0 = None
        static_castx = np.array([self.xintval[0]])  # unfortunately necessary
        y0 = self.y_from_psix(inpsi = self.psi, xvals_in=static_castx)
        #print(f" y0 = {y0}")
        positions = [ [] for _ in range(len(y0)) ]
        maxlength = 0
        for i in range(len(y0)):
            itercounter = 0
            
            u = (self.xintval[0] + 1j*y0[i])[0]
            (positions[i]).append(u)
            while (np.real(u) < self.xintval[1]):
                velocity = 0 + 0j
                itercounter += 1
                #if itercounter % 2500 == 0:
                #    print(f" particle {y0[i]} at {itercounter}: {np.real(u), np.imag(u)}")

                if self.schema == "FE":
                    velocity = self._particle_velocityfield(u)
                    
                
                elif self.schema == "RK4":
                    
                    k1 = self._particle_velocityfield(u)
                    k2 = self._particle_velocityfield(u + dt*k1/2)
                    k3 = self._particle_velocityfield(u + dt*k2/2)
                    k4 = self._particle_velocityfield(u + dt*k3)
                    velocity = (k1+2*k2+2*k3+k4)/6
                if (np.real(velocity) < 0):
                    logger.warning(f" BAD VELOCITY AT y0 = {y0}, step {itercounter}")
                    quit()
                u += velocity*dt
                if np.abs(np.real((positions[i])[-1] - u)) > mindx:
                    # if this size of dt renders oversized dx
                    # reduce timestep size, and then reset the step
                    dt /= 2
                    u = (positions[i])[-1]
                else:
                    (positions[i]).append(u)

        maxlength = max(len(i) for i in positions)            
        pos_array = np.empty((len(y0), maxlength), dtype=complex)
        print(f" maxlen {maxlength}")
        for i in range(len(pos_array)):
            ulen = len(positions[i])
            pos_array[i,:ulen] = np.array(positions[i])
            pos_array[i,ulen:] = np.nan
        return pos_array


    def evaluate_xerror(self):
        """
        1. calculer la trajectoires des particules avec le schema choisi

        Puis, pour chaque trajectoire:
        1. on calcule psi
        2. on isole les valeurs reelles (x)
        3. on retrouve la trajectoire analytique avec .y_from_psix(psi, x)
            -- exactement la solution voulue
        4. Pour trouver l'integrale on fait la partition de la trajectoire
            comme elle se présente directement
        """
        self.schema = "RK4"

        rk4 = self.xintval_integration()
        validity_vec = np.argmin(np.isnan(rk4) == False, axis=1)
        psivals = self.v0*(1-self.a**2/ ((np.abs(rk4[:,0]))**2) )*np.imag(rk4[:,0])
        true_rk4 = np.zeros(rk4.shape, dtype=complex)
        yerrors = np.zeros(len(rk4))
        for i in range(len(rk4)):
            last_idx = 0
            if validity_vec[i] != 0:
                last_idx = validity_vec[i]
            else:
                last_idx = -1
            true_rk4[i,:last_idx] = 1j*self.y_from_psix(np.array([psivals[i]]), np.real(rk4[i,:last_idx]))
            true_rk4[i,:last_idx] += np.real(rk4[i,:last_idx])
            xmaillage = np.real(rk4[i,:last_idx]) - np.real(np.roll(rk4[i, :last_idx], shift=1))
            xmaillage[0] = (self.xintval[1] - self.xintval[0]) - np.sum(xmaillage[1:])
            yerr = xmaillage*np.abs(np.imag(true_rk4[i,:last_idx]) - np.imag(rk4[i,:last_idx]))
            yerrors[i] = np.sum(yerr)
        return yerrors

        #print(rk4[:,0] - true_rk4[:,0])



def main():
    fig, ax = plt.subplots(1)
    flow = SpinGen(gamma=13.5)
    flow.populate_equilines()
    #for psilines in flow.streams:
        #ax.plot(np.real(psilines), np.imag(psilines), color="tab:blue")
    for j in range(len(flow.equiphi)):
        philine = flow.equiphi[j]
        phi = flow.phi[j]
        dx = np.real(philine[1] - philine[0])
        segs = np.append(np.array([1]), np.where(np.abs(np.real(philine[1:]-philine[:-1]))>2*dx))
        segs = np.append(segs, np.array([-1]))
        for i in range(1, len(segs)):
            start, stop = segs[i-1]+1, segs[i]
            ax.plot(np.real(philine[start:stop]), 0.001*j+np.imag(philine[start:stop]), label=phi, marker="x", color="C"+str(j))

    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.sin(t), np.cos(t), color="black")
    ax.set_aspect(1)
    plt.show()

#%%
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=("Flowgen"))
    parser.add_argument("-v", type=int, default=3,
                        help=("set logging level: 0 critical, 1 error, "
                              "2 warning, 3 info, 4 debug, default info"))
    
    args = parser.parse_args()
    logging_translate = [logging.CRITICAL, logging.ERROR, logging.WARNING,
                         logging.INFO, logging.DEBUG]
    logging.basicConfig(level=logging_translate[args.v],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
# %%
